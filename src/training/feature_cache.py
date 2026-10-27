import torch
import pandas as pd

from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.datasets.seasonet import SeasoNetDataset
from src.models.model import SeasonAgnosticModel
from src.models.preprocessing import preprocess_images
from src.preprocessing.schema import Pairs


SPLITS = ("train", "val", "test")
CACHE_SPLITS = ("train", "val")
SEASONS = ("summer", "snow")


def feature_cache_path(artifacts_dir: Path, backbone: str, pairs_path: Path) -> Path:
    return artifacts_dir / f"features_{backbone}_{Path(pairs_path).stem}.pt"


def patch_feature_cache_path(artifacts_dir: Path, backbone: str, pairs_path: Path) -> Path:
    return artifacts_dir / f"patch_features_{backbone}_{Path(pairs_path).stem}.pt"


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
def extract_split_patch_features(
    backbone: nn.Module,
    dataset: SeasoNetDataset,
    device: torch.device,
    image_size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    batch_size: int,
    num_workers: int,
) -> dict:
    if not hasattr(backbone, "forward_features"):
        raise ValueError("Patch feature caching requires a backbone with forward_features")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    def extract(images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        images = preprocess_images(images.to(device), image_size, mean, std)
        features = backbone.forward_features(images)
        try:
            cls_features = features["x_norm_clstoken"]
            patch_features = features["x_norm_patchtokens"]
        except KeyError as error:
            raise ValueError("Backbone forward_features must return CLS and patch tokens") from error
        return cls_features.to(dtype=torch.float16, device="cpu"), patch_features.to(
            dtype=torch.float16,
            device="cpu",
        )

    summer_cls, snow_cls = [], []
    summer_patches, snow_patches = [], []
    for batch in tqdm(loader, desc="Extracting patch features", leave=False):
        summer_cls_features, summer_patch_features = extract(batch.anchor)
        snow_cls_features, snow_patch_features = extract(batch.positive)
        summer_cls.append(summer_cls_features)
        snow_cls.append(snow_cls_features)
        summer_patches.append(summer_patch_features)
        snow_patches.append(snow_patch_features)

    return {
        Pairs.LOC_ID: dataset.pairs[Pairs.LOC_ID].tolist(),
        "summer": {"cls": torch.cat(summer_cls), "patches": torch.cat(summer_patches)},
        "snow": {"cls": torch.cat(snow_cls), "patches": torch.cat(snow_patches)},
    }


def load_feature_cache(path: Path, pairs: pd.DataFrame) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Feature cache not found: {path}. Run scripts/extract_features.py first.")

    cache = torch.load(path, weights_only=False)
    for split in SPLITS:
        expected = pairs.loc[pairs[Pairs.SPLIT] == split, Pairs.LOC_ID].astype(str).tolist()
        if cache["splits"][split][Pairs.LOC_ID] != expected:
            raise ValueError(
                f"Feature cache {path.name} does not match the current pairs table "
                f"({split} split). Re-run scripts/extract_features.py."
            )
    return cache


def load_patch_feature_cache(
    path: Path,
    pairs: pd.DataFrame,
    backbone: str,
    weights: str,
    image_size: int,
    normalization: dict[str, list[float]],
) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"Patch feature cache not found: {path}. Run scripts/extract_patch_features.py first."
        )

    cache = torch.load(path, weights_only=False, mmap=True)
    expected_metadata = {
        "backbone": backbone,
        "weights": weights,
        "image_size": image_size,
        "normalization": normalization,
    }
    for key, expected_value in expected_metadata.items():
        if cache.get(key) != expected_value:
            raise ValueError(
                f"Patch feature cache {path.name} has {key}={cache.get(key)!r}, "
                f"expected {expected_value!r}. Re-run scripts/extract_patch_features.py."
            )

    for split in CACHE_SPLITS:
        expected_ids = pairs.loc[pairs[Pairs.SPLIT] == split, Pairs.LOC_ID].astype(str).tolist()
        split_features = cache["splits"].get(split)
        if split_features is None or split_features[Pairs.LOC_ID] != expected_ids:
            raise ValueError(
                f"Patch feature cache {path.name} does not match the current {split} split. "
                "Re-run scripts/extract_patch_features.py."
            )
        for season in SEASONS:
            cls_features = split_features[season]["cls"]
            patch_features = split_features[season]["patches"]
            if cls_features.ndim != 2 or patch_features.ndim != 3:
                raise ValueError(f"Patch feature cache {path.name} has invalid {split}/{season} tensors")
            if len(cls_features) != len(patch_features) or cls_features.shape[-1] != patch_features.shape[-1]:
                raise ValueError(f"Patch feature cache {path.name} has incompatible {split}/{season} tensors")
    return cache


def forward_heads(
    model: SeasonAgnosticModel,
    cls_features: torch.Tensor,
    grl_scale: float,
    patch_features: torch.Tensor | None = None,
    return_attention: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    # Mirrors SeasonAgnosticModel.forward_training after the backbone.
    return model.forward_feature_training(
        cls_features,
        patch_features,
        grl_scale,
        return_attention=return_attention,
    )


def embed_features(
    model: SeasonAgnosticModel,
    cls_features: torch.Tensor,
    patch_features: torch.Tensor | None = None,
) -> torch.Tensor:
    # Mirrors SeasonAgnosticModel.forward_embeddings after the backbone.
    return model.embed_features(cls_features, patch_features)
