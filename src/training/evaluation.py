import torch

from collections.abc import Callable
from torch.nn import functional as F
from torch.utils.data import DataLoader

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
