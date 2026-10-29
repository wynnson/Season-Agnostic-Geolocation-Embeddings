import math

import torch
from torch import nn


class PatchAttentionPool(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        residual_scale_initialization: float,
        num_regions: int = 1,
    ):
        super().__init__()
        if hidden_dim < 1:
            raise ValueError("Patch-attention hidden_dim must be positive")
        if residual_scale_initialization <= 0:
            raise ValueError("Patch-attention residual scale initialization must be positive")
        if num_regions < 1 or math.isqrt(num_regions) ** 2 != num_regions:
            raise ValueError("Patch-attention num_regions must be a positive square number")

        self.num_regions = num_regions
        self.region_side = math.isqrt(num_regions)
        self.scorer = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_regions),
        )
        self.fusion = nn.Linear(embedding_dim * num_regions, embedding_dim, bias=False)
        self.residual_scale = nn.Parameter(torch.tensor(residual_scale_initialization))

    def _region_mask(self, patch_count: int, device: torch.device) -> torch.Tensor:
        patch_side = math.isqrt(patch_count)
        if patch_side * patch_side != patch_count or patch_side % self.region_side:
            raise ValueError(
                "Patch count must form a square grid divisible by the patch-attention region grid"
            )

        rows = torch.arange(patch_side, device=device)[:, None]
        columns = torch.arange(patch_side, device=device)[None, :]
        region_width = patch_side // self.region_side
        region_ids = (rows // region_width) * self.region_side + columns // region_width
        return torch.nn.functional.one_hot(region_ids.flatten(), self.num_regions).transpose(0, 1).bool()

    def forward(
        self,
        cls_features: torch.Tensor,
        patch_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if cls_features.shape[:-1] != patch_features.shape[:-2]:
            raise ValueError("CLS and patch features must have matching batch dimensions")
        if cls_features.shape[-1] != patch_features.shape[-1]:
            raise ValueError("CLS and patch features must have matching embedding dimensions")

        logits = self.scorer(patch_features).transpose(-2, -1)
        if self.num_regions == 1:
            attention = torch.softmax(logits.squeeze(-2), dim=-1)
            pooled_patches = (attention.unsqueeze(-1) * patch_features).sum(dim=-2)
            fused_features = cls_features + self.residual_scale * self.fusion(pooled_patches)
            return fused_features, attention

        region_mask = self._region_mask(patch_features.shape[-2], patch_features.device)
        attention = torch.softmax(logits.masked_fill(~region_mask, -torch.inf), dim=-1)
        pooled_patches = torch.einsum("brn,bnd->brd", attention, patch_features).flatten(start_dim=-2)
        fused_features = cls_features + self.residual_scale * self.fusion(pooled_patches)
        return fused_features, attention
