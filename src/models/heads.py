import torch
from torch import nn


class ResidualProjectionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float):
        super().__init__()
        if output_dim != input_dim:
            raise ValueError(
                "Residual projection heads require embedding_dim to match the backbone embedding dimension"
            )
        self.normalization = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features + self.mlp(self.normalization(features))


def build_projection_head(
    input_dim: int,
    hidden_dim: int,
    embedding_dim: int,
    dropout: float,
    architecture: str,
) -> nn.Module:
    match architecture:
        case "mlp":
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, embedding_dim),
            )
        case "residual":
            return ResidualProjectionHead(input_dim, hidden_dim, embedding_dim, dropout)
        case _:
            raise ValueError(f"Unsupported projection head architecture: {architecture}")
