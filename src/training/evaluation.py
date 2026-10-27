import torch

from collections.abc import Callable
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.training.patch_reranking import (
    recall_at_k_from_candidates,
    rerank_patch_candidates,
    retrieve_top_candidates,
)
from src.training.schema import ValidationMetrics


@torch.inference_mode()
def retrieval_recall_at_k(
    queries: torch.Tensor,
    gallery: torch.Tensor,
    ks: tuple[int, ...] = (1, 5),
    chunk_size: int = 512,
) -> dict[str, float]:
    if queries.shape != gallery.shape:
        raise ValueError("Queries and gallery must have matching shapes")

    if not ks or min(ks) < 1 or chunk_size < 1 or len(queries) == 0:
        raise ValueError("Invalid ks, chunk_size, or empty embeddings")

    # normalize embeddings for cosine similarity
    queries = F.normalize(queries, dim=1)
    gallery = F.normalize(gallery, dim=1)

    num_queries = len(queries)
    max_k = min(max(ks), len(gallery))
    hits = {k: 0 for k in ks}

    for start in range(0, num_queries, chunk_size):
        end = min(start + chunk_size, num_queries)

        similarities = queries[start: end] @ gallery.T          # find similar embeddings
        top_indices = similarities.topk(max_k, dim=1).indices   # take top k

        # ground truth
        current_embedding_idx = torch.arange(
            start, end, device=queries.device
        ).unsqueeze(1)

        for k in ks:
            retrieved = top_indices[:, :k]
            correct = (retrieved == current_embedding_idx).any(dim=1)
            hits[k] += correct.sum().item()

    return {
        f"recall@{k}": hits[k] / num_queries
        for k in ks
    }


def build_embeddings_list(
    embed_fn: Callable[[torch.Tensor], torch.Tensor],
    data_loader: DataLoader,
    log_every: int
):
    summer_embeddings = []
    snow_embeddings = []
    processed_counter = 0

    for batch_index, batch in enumerate(data_loader, start=1):
        summer_embeddings.append(embed_fn(batch.anchor))
        snow_embeddings.append(embed_fn(batch.positive))
        processed_counter += len(batch.loc_id)

        if batch_index % log_every == 0 or batch_index == len(data_loader):
            print(f"Embedded {processed_counter:,}/{len(data_loader.dataset):,} evaluation pairs", flush=True)

    summer_embeddings = torch.cat(summer_embeddings)
    snow_embeddings = torch.cat(snow_embeddings)
    return summer_embeddings, snow_embeddings


def build_embeddings_and_patch_tokens(
    embed_fn: Callable[[torch.Tensor], torch.Tensor],
    patch_embed_fn: Callable[[torch.Tensor], torch.Tensor],
    data_loader: DataLoader,
    log_every: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    summer_embeddings, snow_embeddings = [], []
    summer_patches, snow_patches = [], []
    processed_counter = 0

    for batch_index, batch in enumerate(data_loader, start=1):
        summer_embeddings.append(embed_fn(batch.anchor))
        snow_embeddings.append(embed_fn(batch.positive))
        summer_patches.append(patch_embed_fn(batch.anchor).to(device="cpu", dtype=torch.float16))
        snow_patches.append(patch_embed_fn(batch.positive).to(device="cpu", dtype=torch.float16))
        processed_counter += len(batch.loc_id)

        if batch_index % log_every == 0 or batch_index == len(data_loader):
            print(f"Embedded {processed_counter:,}/{len(data_loader.dataset):,} evaluation pairs", flush=True)

    return (
        torch.cat(summer_embeddings),
        torch.cat(snow_embeddings),
        torch.cat(summer_patches),
        torch.cat(snow_patches),
    )


@torch.inference_mode()
def evaluate_cross_season_retrieval(
    data_loader: DataLoader,
    embed_fn: Callable[[torch.Tensor], torch.Tensor],
    chunk_size: int,
    log_every: int,
) -> ValidationMetrics:
    summer_embeddings, snow_embeddings = build_embeddings_list(embed_fn, data_loader, log_every)

    return ValidationMetrics(
        summer_to_snow=retrieval_recall_at_k(
            queries=summer_embeddings,
            gallery=snow_embeddings,
            chunk_size=chunk_size,
        ),
        snow_to_summer=retrieval_recall_at_k(
            queries=snow_embeddings,
            gallery=summer_embeddings,
            chunk_size=chunk_size,
        ),
    )


@torch.inference_mode()
def evaluate_patch_reranked_retrieval(
    summer_embeddings: torch.Tensor,
    snow_embeddings: torch.Tensor,
    summer_patches: torch.Tensor,
    snow_patches: torch.Tensor,
    top_k: int,
    alpha: float,
    retrieval_chunk_size: int,
    reranking_batch_size: int,
    candidate_batch_size: int,
) -> tuple[ValidationMetrics, ValidationMetrics]:
    if top_k < 5:
        raise ValueError("Patch reranking requires top_k to be at least 5")

    def rerank(
        queries: torch.Tensor,
        gallery: torch.Tensor,
        query_patches: torch.Tensor,
        gallery_patches: torch.Tensor,
    ) -> tuple[dict[str, float], dict[str, float]]:
        global_scores, candidate_indices = retrieve_top_candidates(
            queries, gallery, top_k, retrieval_chunk_size
        )
        global_candidate_recall = recall_at_k_from_candidates(candidate_indices, (1, 5, top_k))
        _, reranked_indices = rerank_patch_candidates(
            global_scores,
            candidate_indices,
            query_patches,
            gallery_patches,
            alpha,
            reranking_batch_size,
            candidate_batch_size,
        )
        reranked_recall = recall_at_k_from_candidates(reranked_indices, (1, 5))
        return reranked_recall, global_candidate_recall

    reranked_summer_to_snow, global_candidate_summer_to_snow = rerank(
        summer_embeddings, snow_embeddings, summer_patches, snow_patches
    )
    reranked_snow_to_summer, global_candidate_snow_to_summer = rerank(
        snow_embeddings, summer_embeddings, snow_patches, summer_patches
    )
    return (
        ValidationMetrics(
            summer_to_snow=reranked_summer_to_snow,
            snow_to_summer=reranked_snow_to_summer,
        ),
        ValidationMetrics(
            summer_to_snow=global_candidate_summer_to_snow,
            snow_to_summer=global_candidate_snow_to_summer,
        ),
    )
