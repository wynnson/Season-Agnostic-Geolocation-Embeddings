import torch
from torch import nn


class PatchAttentionPool(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        residual_scale_initialization: float,
    ):
        super().__init__()
        if hidden_dim < 1:
            raise ValueError("Patch-attention hidden_dim must be positive")
        if residual_scale_initialization <= 0:
            raise ValueError("Patch-attention residual scale initialization must be positive")

        self.scorer = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.fusion = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.residual_scale = nn.Parameter(torch.tensor(residual_scale_initialization))

    def forward(
        self,
        cls_features: torch.Tensor,
        patch_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if cls_features.shape[:-1] != patch_features.shape[:-2]:
            raise ValueError("CLS and patch features must have matching batch dimensions")
        if cls_features.shape[-1] != patch_features.shape[-1]:
            raise ValueError("CLS and patch features must have matching embedding dimensions")

        attention = torch.softmax(self.scorer(patch_features).squeeze(-1), dim=-1)
        pooled_patches = (attention.unsqueeze(-1) * patch_features).sum(dim=-2)
        fused_features = cls_features + self.residual_scale * self.fusion(pooled_patches)
        return fused_features, attention
