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


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be positive")
        if alpha <= 0:
            raise ValueError("LoRA alpha must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("LoRA dropout must be in [0, 1)")

        self.base = base
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)

        for parameter in self.base.parameters():
            parameter.requires_grad = False

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    @property
    def weight(self) -> nn.Parameter:
        return self.base.weight

    @property
    def bias(self) -> nn.Parameter | None:
        return self.base.bias

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        update = F.linear(F.linear(self.dropout(inputs), self.lora_a), self.lora_b)
        return self.base(inputs) + self.scale * update


def apply_lora(
    backbone: nn.Module,
    rank: int,
    alpha: float,
    dropout: float,
    target_modules: list[str],
) -> int:
    if not target_modules:
        raise ValueError("LoRA target_modules must not be empty")

    for parameter in backbone.parameters():
        parameter.requires_grad = False

    targets = set(target_modules)
    replaced = 0
    for name, module in list(backbone.named_modules()):
        if not name or not isinstance(module, nn.Linear):
            continue

        leaf_name = name.rsplit(".", maxsplit=1)[-1]
        if name not in targets and leaf_name not in targets:
            continue

        parent_name, _, child_name = name.rpartition(".")
        parent = backbone if not parent_name else backbone.get_submodule(parent_name)
        parent._modules[child_name] = LoRALinear(module, rank, alpha, dropout)
        replaced += 1

    if replaced == 0:
        raise ValueError(
            f"LoRA targets {sorted(targets)} did not match any linear backbone modules"
        )

    return replaced


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
    lora_config=None,
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
        if lora_config is None:
            raise ValueError("LoRA configuration is required when model.tuning is 'lora'")
        adapter_count = apply_lora(
            backbone,
            rank=lora_config.rank,
            alpha=lora_config.alpha,
            dropout=lora_config.dropout,
            target_modules=list(lora_config.target_modules),
        )
        print(f"Using LoRA adapters in {adapter_count} backbone linear layers")
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
