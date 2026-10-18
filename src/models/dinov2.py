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


class SeasonAgnosticDINOv2(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        embedding_dim: int,
        hidden_dim: int,
        projection_dropout: float,
        season_classes: int,
    ):
        super().__init__()
        self.backbone = backbone
        backbone_dim = getattr(backbone, "embed_dim", None)
        if not isinstance(backbone_dim, int):
            raise ValueError("DINOv2 backbone must expose an integer embed_dim")

        self.projection_head = nn.Sequential(
            nn.Linear(backbone_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(projection_dropout),
            nn.Linear(hidden_dim, embedding_dim),
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


def load_dinov2(
    model_name: str,
    repository: str,
    tuning: str,
    embedding_dim: int,
    hidden_dim: int,
    projection_dropout: float,
    season_classes: int,
) -> SeasonAgnosticDINOv2:
    backbone = torch.hub.load(repository, model_name)
    model = SeasonAgnosticDINOv2(
        backbone=backbone,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        projection_dropout=projection_dropout,
        season_classes=season_classes,
    )

    if tuning == "frozen":
        model.freeze_backbone()
    elif tuning == "lora":
        raise NotImplementedError("LoRA adapter insertion is not implemented yet")
    elif tuning != "full":
        raise ValueError(f"Unsupported DINOv2 tuning mode: {tuning}")

    return model


def load_frozen_dinov2(model_name: str, repository: str) -> nn.Module:
    model = torch.hub.load(repository, model_name)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


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


def embed_dinov2_images(
    model: nn.Module,
    images: torch.Tensor,
    image_size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> torch.Tensor:
    images = preprocess_images(images, image_size, mean, std)
    if isinstance(model, SeasonAgnosticDINOv2):
        return model.forward_embeddings(images)
    return F.normalize(model(images), dim=1)
