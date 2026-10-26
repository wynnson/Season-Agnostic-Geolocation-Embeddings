import argparse
from pathlib import Path
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
    SPLITS,
    extract_split_features,
    extract_split_intermediate_features,
    feature_cache_path,
    intermediate_feature_cache_path,
)
from src.utils.device import get_device


EXTRACTION_BATCH_SIZE = 128


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Cache frozen backbone features.")
    parser.add_argument(
        "--intermediate",
        action="store_true",
        help="Cache CLS features from every transformer block.",
    )
    return parser.parse_known_args()


def main() -> None:
    args, overrides = parse_args()
    cfg = OmegaConf.merge(
        OmegaConf.load(REPO_DIR / "config/train_config.yaml"),
        OmegaConf.from_dotlist(overrides),
    )
    preprocessing_cfg = OmegaConf.load(REPO_DIR / "config/preprocessing_config.yaml")
    artifacts_dir = REPO_DIR / preprocessing_cfg.output.artifacts_dir
    manifest = OmegaConf.load(artifacts_dir / preprocessing_cfg.output.manifest_filename)

    data_dir = (REPO_DIR / cfg.data.data_dir).resolve()
    pairs = pd.read_parquet(REPO_DIR / manifest.pairs_path)

    device = get_device()
    backbone = load_frozen_backbone(
        cfg.model.backbone,
        cfg.model.repository,
        cfg.model.weights,
    ).to(device)

    layers = None
    if args.intermediate:
        blocks = getattr(backbone, "blocks", None)
        if blocks is None:
            raise ValueError("Backbone does not expose transformer blocks")
        layers = list(range(len(blocks)))
        print(f"Caching CLS embeddings from layers: {layers}", flush=True)

    splits = {}
    for split in SPLITS:
        dataset = SeasoNetDataset(pairs, split=split, data_dir=data_dir)
        print(f"Extracting {split}: {len(dataset):,} pairs", flush=True)
        if layers is None:
            splits[split] = extract_split_features(
                backbone,
                dataset,
                device=device,
                image_size=cfg.model.image_size,
                mean=tuple(cfg.model.normalization.mean),
                std=tuple(cfg.model.normalization.std),
                batch_size=EXTRACTION_BATCH_SIZE,
                num_workers=cfg.data.num_workers,
            )
        else:
            splits[split] = extract_split_intermediate_features(
                backbone,
                dataset,
                device=device,
                image_size=cfg.model.image_size,
                mean=tuple(cfg.model.normalization.mean),
                std=tuple(cfg.model.normalization.std),
                batch_size=EXTRACTION_BATCH_SIZE,
                num_workers=cfg.data.num_workers,
                layers=layers,
            )

    if layers is None:
        output_path = feature_cache_path(artifacts_dir, cfg.model.backbone, Path(manifest.pairs_path))
        cache = {
            "backbone": cfg.model.backbone,
            "image_size": cfg.model.image_size,
            "pairs_path": manifest.pairs_path,
            "splits": splits,
        }
        label = "features"
    else:
        output_path = intermediate_feature_cache_path(
            artifacts_dir,
            cfg.model.backbone,
            layers,
            Path(manifest.pairs_path),
        )
        cache = {
            "backbone": cfg.model.backbone,
            "image_size": cfg.model.image_size,
            "layers": layers,
            "pairs_path": manifest.pairs_path,
            "splits": splits,
        }
        label = "intermediate features"

    torch.save(cache, output_path)
    print(f"Saved {label}: {output_path.relative_to(REPO_DIR)} ({output_path.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
