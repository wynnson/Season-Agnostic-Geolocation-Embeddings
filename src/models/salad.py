import math

import torch
from torch import nn
from torch.nn import functional as F


def sinkhorn_assignments(
    scores: torch.Tensor,
    dustbin_score: torch.Tensor,
    iterations: int,
) -> torch.Tensor:
    """Return balanced patch-to-cluster assignments with a dustbin cluster."""
    batch_size, clusters, patches = scores.shape
    if patches <= clusters:
        raise ValueError("SALAD requires more patches than clusters")

    augmented = torch.cat(
        (scores, dustbin_score.expand(batch_size, 1, patches)),
        dim=1,
    )
    normalizer = -math.log(patches + clusters)
    log_clusters = scores.new_full((batch_size, clusters + 1), normalizer)
    log_clusters[:, -1] += math.log(patches - clusters)
    log_patches = scores.new_full((batch_size, patches), normalizer)
    u = torch.zeros_like(log_clusters)
    v = torch.zeros_like(log_patches)

    for _ in range(iterations):
        u = log_clusters - torch.logsumexp(augmented + v.unsqueeze(1), dim=2)
        v = log_patches - torch.logsumexp(augmented + u.unsqueeze(2), dim=1)

    return (augmented + u.unsqueeze(2) + v.unsqueeze(1) - normalizer).exp()


class SALADPool(nn.Module):
    """Optimal-transport aggregation of patch descriptors plus a global CLS token."""

    def __init__(
        self,
        input_dim: int,
        num_clusters: int,
        cluster_dim: int,
        token_dim: int,
        hidden_dim: int,
        dropout: float,
        sinkhorn_iterations: int,
    ):
        super().__init__()
        if min(num_clusters, cluster_dim, token_dim, hidden_dim, sinkhorn_iterations) < 1:
            raise ValueError("SALAD dimensions and Sinkhorn iterations must be positive")

        self.num_clusters = num_clusters
        self.cluster_dim = cluster_dim
        self.token_dim = token_dim
        self.sinkhorn_iterations = sinkhorn_iterations
        regularizer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.token_features = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, token_dim),
        )
        self.cluster_features = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            regularizer,
            nn.ReLU(),
            nn.Linear(hidden_dim, cluster_dim),
        )
        self.scores = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_clusters),
        )
        self.dustbin_score = nn.Parameter(torch.tensor(1.0))

    @property
    def output_dim(self) -> int:
        return self.num_clusters * self.cluster_dim + self.token_dim

    def forward(self, cls_features: torch.Tensor, patch_features: torch.Tensor) -> torch.Tensor:
        if cls_features.ndim != 2 or patch_features.ndim != 3:
            raise ValueError("SALAD expects CLS [B, D] and patch features [B, N, D]")
        if cls_features.shape[0] != patch_features.shape[0]:
            raise ValueError("CLS and patch feature batches must match")

        local_features = self.cluster_features(patch_features)
        scores = self.scores(patch_features).transpose(1, 2)
        assignments = sinkhorn_assignments(
            scores,
            self.dustbin_score,
            self.sinkhorn_iterations,
        )[:, :self.num_clusters]
        aggregated = torch.einsum("bmn,bnd->bmd", assignments, local_features)
        aggregated = F.normalize(aggregated, dim=-1).flatten(1)
        token = F.normalize(self.token_features(cls_features), dim=-1)
        return F.normalize(torch.cat((token, aggregated), dim=-1), dim=-1)
