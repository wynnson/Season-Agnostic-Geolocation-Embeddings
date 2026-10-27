import math
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
    patch_attention_config=None,
) -> SeasonAgnosticModel:
    backbone = load_backbone(model_name, repository, weights)
    model = SeasonAgnosticModel(
        backbone=backbone,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        projection_dropout=projection_dropout,
        season_classes=season_classes,
        projection_architecture=projection_architecture,
        patch_attention_config=patch_attention_config,
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


def first_lora_block(target_modules: list[str]) -> int:
    blocks = set()
    for name in target_modules:
        parts = name.split(".")
        if len(parts) < 3 or parts[0] != "blocks" or not parts[1].isdigit():
            raise ValueError(f"LoRA target {name!r} must name a block module, such as blocks.10.attn.qkv")
        blocks.add(int(parts[1]))
    return min(blocks)


def _rope(backbone: nn.Module, height: int, width: int):
    if not hasattr(backbone, "rope_embed"):
        raise ValueError("Block-level token caching supports DINOv3 backbones only")
    return None if backbone.rope_embed is None else backbone.rope_embed(H=height, W=width)


def tokens_before_block(backbone: nn.Module, images: torch.Tensor, block: int) -> torch.Tensor:
    tokens, (height, width) = backbone.prepare_tokens_with_masks(images)
    rope = _rope(backbone, height, width)
    for layer in backbone.blocks[:block]:
        tokens = layer(tokens, rope)
    return tokens


def forward_from_block(backbone: nn.Module, tokens: torch.Tensor, block: int) -> torch.Tensor:
    # Mirrors DINOv3's forward from `block` on: remaining blocks, CLS norm, head.
    side = math.isqrt(tokens.shape[1] - 1 - backbone.n_storage_tokens)
    rope = _rope(backbone, side, side)
    for layer in backbone.blocks[block:]:
        tokens = layer(tokens, rope)
    norm = backbone.norm if backbone.cls_norm is None else backbone.cls_norm
    return backbone.head(norm(tokens[:, 0]))


def load_frozen_backbone(
    model_name: str, repository: str, weights: str | Path | None = None
) -> nn.Module:
    backbone = load_backbone(model_name, repository, weights)
    backbone.eval()

    for parameter in backbone.parameters():
        parameter.requires_grad = False

    return backbone
