import torch
from torch import nn


class LayerMixer(nn.Module):
    """Learn a convex combination of intermediate transformer-layer features."""

    def __init__(self, layers: list[int]):
        super().__init__()
        if not layers:
            raise ValueError("Layer mixer requires at least one layer")
        if len(set(layers)) != len(layers):
            raise ValueError("Layer mixer layers must be unique")

        self.layers = tuple(layers)
        # Equal logits produce an unbiased 1 / num_layers initial mixture.
        self.logits = nn.Parameter(torch.zeros(len(layers)))

    def weights(self) -> torch.Tensor:
        return torch.softmax(self.logits, dim=0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim < 3:
            raise ValueError("Layer mixer expects features shaped [..., layers, embedding_dim]")
        if features.shape[-2] != len(self.layers):
            raise ValueError(
                f"Layer mixer expected {len(self.layers)} layers, got {features.shape[-2]}"
            )
        return torch.einsum("...ld,l->...d", features, self.weights())
