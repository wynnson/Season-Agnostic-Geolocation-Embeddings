import torch
from torch.nn import functional as F


def retrieve_top_candidates(
    queries: torch.Tensor,
    gallery: torch.Tensor,
    top_k: int,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if queries.ndim != 2 or gallery.ndim != 2 or queries.shape[1] != gallery.shape[1]:
        raise ValueError("Queries and gallery must be 2D tensors with matching embedding dimensions")
    if not 1 <= top_k <= len(gallery):
        raise ValueError("top_k must be between 1 and the gallery size")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    queries = F.normalize(queries, dim=-1)
    gallery = F.normalize(gallery, dim=-1)
    scores, indices = [], []
    for start in range(0, len(queries), chunk_size):
        similarities = queries[start:start + chunk_size] @ gallery.T
        chunk_scores, chunk_indices = similarities.topk(top_k, dim=1)
        scores.append(chunk_scores)
        indices.append(chunk_indices)
    return torch.cat(scores), torch.cat(indices)


def corresponding_patch_similarity(
    queries: torch.Tensor,
    candidates: torch.Tensor,
) -> torch.Tensor:
    """Return mean cosine similarity between corresponding patch positions.

    Queries are [batch, patches, dim] and candidates are
    [batch, candidates, patches, dim].
    """
    if queries.ndim != 3 or candidates.ndim != 4:
        raise ValueError("Expected query [batch, patches, dim] and candidate [batch, candidates, patches, dim]")
    if queries.shape[0] != candidates.shape[0] or queries.shape[1:] != candidates.shape[2:]:
        raise ValueError("Query and candidate patch shapes must match")
    queries = F.normalize(queries, dim=-1)
    candidates = F.normalize(candidates, dim=-1)
    return (queries[:, None] * candidates).sum(dim=-1).mean(dim=-1)


@torch.inference_mode()
def rerank_patch_candidates(
    global_scores: torch.Tensor,
    candidate_indices: torch.Tensor,
    query_patches: torch.Tensor,
    gallery_patches: torch.Tensor,
    alpha: float,
    batch_size: int,
    candidate_batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    if global_scores.shape != candidate_indices.shape:
        raise ValueError("Global scores and candidate indices must have matching shapes")
    if len(query_patches) != len(global_scores):
        raise ValueError("Each query must have patch tokens")
    if batch_size < 1 or candidate_batch_size < 1:
        raise ValueError("Reranking batch sizes must be positive")

    device = global_scores.device
    reranked_scores = torch.empty_like(global_scores)
    for start in range(0, len(global_scores), batch_size):
        end = min(start + batch_size, len(global_scores))
        query_batch = query_patches[start:end].to(device=device, dtype=torch.float32)
        for candidate_start in range(0, global_scores.shape[1], candidate_batch_size):
            candidate_end = min(candidate_start + candidate_batch_size, global_scores.shape[1])
            indices = candidate_indices[start:end, candidate_start:candidate_end]
            candidate_batch = gallery_patches[indices.cpu()].to(device=device, dtype=torch.float32)
            patch_scores = corresponding_patch_similarity(query_batch, candidate_batch)
            reranked_scores[start:end, candidate_start:candidate_end] = (
                (1 - alpha) * global_scores[start:end, candidate_start:candidate_end]
                + alpha * patch_scores
            )

    ranks = reranked_scores.argsort(dim=1, descending=True)
    return reranked_scores.gather(1, ranks), candidate_indices.gather(1, ranks)


def recall_at_k_from_candidates(candidate_indices: torch.Tensor, ks: tuple[int, ...]) -> dict[str, float]:
    if candidate_indices.ndim != 2 or not ks or min(ks) < 1:
        raise ValueError("Invalid candidate indices or recall values")

    expected = torch.arange(len(candidate_indices), device=candidate_indices.device).unsqueeze(1)
    return {
        f"recall@{k}": (candidate_indices[:, :k] == expected).any(dim=1).float().mean().item()
        for k in ks
    }
