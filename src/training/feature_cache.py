import torch
import pandas as pd

from pathlib import Path
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.datasets.seasonet import SeasoNetDataset
from src.models.gradient_reversal import gradient_reverse
from src.models.model import SeasonAgnosticModel
from src.models.preprocessing import preprocess_images
from src.preprocessing.schema import Pairs


SPLITS = ("train", "val", "test")
SEASONS = ("summer", "snow")


def feature_cache_path(artifacts_dir: Path, backbone: str, pairs_path: Path) -> Path:
    return artifacts_dir / f"features_{backbone}_{Path(pairs_path).stem}.pt"


def intermediate_feature_cache_path(
    artifacts_dir: Path,
    backbone: str,
    layers: list[int],
    pairs_path: Path,
) -> Path:
    layer_key = "-".join(map(str, layers))
    return artifacts_dir / f"intermediate_features_{backbone}_layers{layer_key}_{Path(pairs_path).stem}.pt"


@torch.inference_mode()
def extract_split_features(
    backbone: nn.Module,
    dataset: SeasoNetDataset,
    device: torch.device,
    image_size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    batch_size: int,
    num_workers: int,
) -> dict:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    def embed(images: torch.Tensor) -> torch.Tensor:
        images = preprocess_images(images.to(device), image_size, mean, std)
        return backbone(images).float().cpu()

    summer, snow = [], []
    for batch in tqdm(loader, desc="Extracting", leave=False):
        summer.append(embed(batch.anchor))
        snow.append(embed(batch.positive))

    return {
        Pairs.LOC_ID: dataset.pairs[Pairs.LOC_ID].tolist(),
        "summer": torch.cat(summer),
        "snow": torch.cat(snow),
    }


@torch.inference_mode()
def extract_split_intermediate_features(
    backbone: nn.Module,
    dataset: SeasoNetDataset,
    device: torch.device,
    image_size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    batch_size: int,
    num_workers: int,
    layers: list[int],
) -> dict:
    if not hasattr(backbone, "get_intermediate_layers"):
        raise ValueError("Backbone does not expose get_intermediate_layers")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    def embed(images: torch.Tensor) -> torch.Tensor:
        images = preprocess_images(images.to(device), image_size, mean, std)
        outputs = backbone.get_intermediate_layers(
            images,
            n=layers,
            return_class_token=True,
        )
        return torch.stack([class_token for _, class_token in outputs], dim=1).float().cpu()

    summer, snow = [], []
    for batch in tqdm(loader, desc="Extracting intermediate layers", leave=False):
        summer.append(embed(batch.anchor))
        snow.append(embed(batch.positive))

    return {
        Pairs.LOC_ID: dataset.pairs[Pairs.LOC_ID].tolist(),
        "summer": torch.cat(summer),
        "snow": torch.cat(snow),
    }


def load_feature_cache(path: Path, pairs: pd.DataFrame, mmap: bool = False) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Feature cache not found: {path}. Run scripts/extract_features.py first.")

    cache = torch.load(path, weights_only=False, mmap=mmap)
    for split in SPLITS:
        expected = pairs.loc[pairs[Pairs.SPLIT] == split, Pairs.LOC_ID].astype(str).tolist()
        if cache["splits"][split][Pairs.LOC_ID] != expected:
            raise ValueError(
                f"Feature cache {path.name} does not match the current pairs table "
                f"({split} split). Re-run scripts/extract_features.py."
            )
    return cache


def load_intermediate_feature_cache(path: Path, pairs: pd.DataFrame, layers: list[int]) -> dict:
    cache = load_feature_cache(path, pairs, mmap=True)
    if cache.get("layers") != layers:
        raise ValueError(
            f"Intermediate cache {path.name} has layers {cache.get('layers')}, expected {layers}. "
            "Re-run scripts/extract_features.py --intermediate."
        )

    for split in SPLITS:
        for season in SEASONS:
            features = cache["splits"][split][season]
            if features.ndim != 3 or features.shape[1] != len(layers):
                raise ValueError(
                    f"Intermediate cache {path.name} has invalid {split}/{season} shape "
                    f"{tuple(features.shape)}."
                )
    return cache


def forward_heads(
    model: SeasonAgnosticModel,
    features: torch.Tensor,
    grl_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Mirrors SeasonAgnosticModel.forward_training after the backbone.
    embedding = F.normalize(model.projection_head(model.mix_features(features)), dim=-1)
    season_logits = model.season_classifier(gradient_reverse(embedding, grl_scale))
    return embedding, season_logits


def embed_features(model: SeasonAgnosticModel, features: torch.Tensor) -> torch.Tensor:
    # Mirrors SeasonAgnosticModel.forward_embeddings after the backbone.
    return F.normalize(model.projection_head(model.mix_features(features)), dim=-1)
