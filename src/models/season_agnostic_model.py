from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import Normalize

class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.scale * grad_output, None


def gradient_reverse(x: torch.Tensor, scale: float) -> torch.Tensor:
    return _GradientReverse.apply(x, scale)


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


class SeasonAgnosticModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        embedding_dim: int,
        hidden_dim: int,
        projection_dropout: float,
        season_classes: int,
        projection_architecture: str = "mlp",
    ):
        super().__init__()
        self.backbone = backbone
        backbone_dim = getattr(backbone, "embed_dim", None)

        if not isinstance(backbone_dim, int):
            raise ValueError("Backbone must expose an integer embed_dim")

        self.projection_head = build_projection_head(
            backbone_dim,
            hidden_dim,
            embedding_dim,
            projection_dropout,
            projection_architecture,
        )
        self.season_classifier = nn.Linear(embedding_dim, season_classes)
        self.backbone_frozen = False

    def forward_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        embedding = self.projection_head(features)
        return F.normalize(embedding, dim=1)

    def forward_training(
        self, images: torch.Tensor, grl_scale: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(images)
        embedding = self.projection_head(features)
        normalized_out = F.normalize(embedding, dim=1)
        season_features = gradient_reverse(normalized_out, grl_scale)
        season_logits = self.season_classifier(season_features)
        return normalized_out, season_logits

    def freeze_backbone(self) -> None:
        self.backbone_frozen = True
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.backbone_frozen:
            self.backbone.eval()
        return self


def load_backbone(model_name: str, repository: str, weights: str | Path | None = None) -> nn.Module:
    # Some backbone weights are gated, so torch.hub needs a local copy.
    if weights is None:
        return torch.hub.load(repository, model_name)

    weights = Path(weights)
    if not weights.is_file():
        raise FileNotFoundError(f"Backbone weights not found: {weights}")

    return torch.hub.load(repository, model_name, weights=str(weights))


def load_model(
    model_name: str,
    repository: str,
    tuning: str,
    embedding_dim: int,
    hidden_dim: int,
    projection_dropout: float,
    season_classes: int,
    projection_architecture: str = "mlp",
    weights: str | Path | None = None,
) -> SeasonAgnosticModel:
    backbone = load_backbone(model_name, repository, weights)
    model = SeasonAgnosticModel(
        backbone=backbone,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        projection_dropout=projection_dropout,
        season_classes=season_classes,
        projection_architecture=projection_architecture,
    )

    if tuning == "frozen":
        model.freeze_backbone()
    elif tuning == "lora":
        raise NotImplementedError("LoRA adapter insertion is not implemented yet")
    elif tuning != "full":
        raise ValueError(f"Unsupported backbone tuning mode: {tuning}")

    print(f"Using: {model_name} backbone")
    return model


def load_frozen_backbone(
    model_name: str, repository: str, weights: str | Path | None = None
) -> nn.Module:
    backbone = load_backbone(model_name, repository, weights)
    backbone.eval()    
    
    for parameter in backbone.parameters():
        parameter.requires_grad = False

    return backbone


def preprocess_images(
    images: torch.Tensor,
    image_size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> torch.Tensor:
    if images.shape[-2:] != (image_size, image_size):
        images = F.interpolate(
            images,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )
    normalize = Normalize(mean=mean, std=std)
    return normalize(images)


def embed_images(
    model: nn.Module,
    images: torch.Tensor,
    image_size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> torch.Tensor:
    images = preprocess_images(images, image_size, mean, std)
    if isinstance(model, SeasonAgnosticModel):
        return model.forward_embeddings(images)
    return F.normalize(model(images), dim=1)
