from pathlib import Path

import torch
from torch import nn

from src.models.lora import apply_lora
from src.models.model import SeasonAgnosticModel


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
    layer_mixer_config=None,
) -> SeasonAgnosticModel:
    backbone = load_backbone(model_name, repository, weights)
    model = SeasonAgnosticModel(
        backbone=backbone,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        projection_dropout=projection_dropout,
        season_classes=season_classes,
        projection_architecture=projection_architecture,
        layer_mixer_config=layer_mixer_config,
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
