import argparse
from pathlib import Path
import shutil
import sys

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import pandas as pd
import torch

from omegaconf import OmegaConf

from src.datasets.seasonet import SeasoNetDataset
from src.models.backbone import load_frozen_backbone
from src.training.feature_cache import (
    CACHE_SPLITS,
    extract_split_patch_features,
    patch_feature_cache_path,
)
from src.utils.device import get_device


EXTRACTION_BATCH_SIZE = 64


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Cache frozen DINO CLS and patch features.")
    return parser.parse_known_args()


def estimated_cache_size_bytes(pairs: pd.DataFrame, embedding_dim: int = 384, patches: int = 196) -> int:
    pair_count = pairs["split"].isin(CACHE_SPLITS).sum()
    values_per_image = embedding_dim + patches * embedding_dim
    return pair_count * 2 * values_per_image * torch.tensor([], dtype=torch.float16).element_size()


def main() -> None:
    _, overrides = parse_args()
    cfg = OmegaConf.merge(
        OmegaConf.load(REPO_DIR / "config/train_config.yaml"),
        OmegaConf.from_dotlist(overrides),
    )
    preprocessing_cfg = OmegaConf.load(REPO_DIR / "config/preprocessing_config.yaml")
    artifacts_dir = REPO_DIR / preprocessing_cfg.output.artifacts_dir
    manifest = OmegaConf.load(artifacts_dir / preprocessing_cfg.output.manifest_filename)
    pairs = pd.read_parquet(REPO_DIR / manifest.pairs_path)

    estimated_bytes = estimated_cache_size_bytes(pairs)
    available_bytes = shutil.disk_usage(artifacts_dir).free
    if estimated_bytes > available_bytes:
        raise RuntimeError(
            f"Patch cache needs approximately {estimated_bytes / 1e9:.1f} GB, but only "
            f"{available_bytes / 1e9:.1f} GB is available."
        )
    print(f"Estimated train/validation patch cache size: {estimated_bytes / 1e9:.1f} GB", flush=True)

    device = get_device()
    backbone = load_frozen_backbone(
        cfg.model.backbone,
        cfg.model.repository,
        cfg.model.weights,
    ).to(device)
    data_dir = (REPO_DIR / cfg.data.data_dir).resolve()

    splits = {}
    for split in CACHE_SPLITS:
        dataset = SeasoNetDataset(pairs, split=split, data_dir=data_dir)
        print(f"Extracting {split}: {len(dataset):,} pairs", flush=True)
        splits[split] = extract_split_patch_features(
            backbone,
            dataset,
            device=device,
            image_size=cfg.model.image_size,
            mean=tuple(cfg.model.normalization.mean),
            std=tuple(cfg.model.normalization.std),
            batch_size=EXTRACTION_BATCH_SIZE,
            num_workers=cfg.data.num_workers,
        )

    output_path = patch_feature_cache_path(artifacts_dir, cfg.model.backbone, Path(manifest.pairs_path))
    torch.save(
        {
            "backbone": cfg.model.backbone,
            "weights": str(cfg.model.weights),
            "image_size": cfg.model.image_size,
            "normalization": OmegaConf.to_container(cfg.model.normalization, resolve=True),
            "pairs_path": manifest.pairs_path,
            "splits": splits,
        },
        output_path,
    )
    print(
        f"Saved patch features: {output_path.relative_to(REPO_DIR)} "
        f"({output_path.stat().st_size / 1e9:.1f} GB)",
        flush=True,
    )


if __name__ == "__main__":
    main()
