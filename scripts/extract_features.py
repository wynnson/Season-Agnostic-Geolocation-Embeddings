from pathlib import Path
import sys

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import pandas as pd
import torch

from omegaconf import OmegaConf

from src.datasets.seasonet import SeasoNetDataset
from src.models.season_agnostic_model import load_frozen_backbone
from src.training.feature_cache import SPLITS, extract_split_features, feature_cache_path
from src.utils.device import get_device


EXTRACTION_BATCH_SIZE = 128


def main() -> None:
    cfg = OmegaConf.merge(
        OmegaConf.load(REPO_DIR / "config/train_config.yaml"),
        OmegaConf.from_dotlist(sys.argv[1:]),
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

    splits = {}
    for split in SPLITS:
        dataset = SeasoNetDataset(pairs, split=split, data_dir=data_dir)
        print(f"Extracting {split}: {len(dataset):,} pairs", flush=True)
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

    output_path = feature_cache_path(artifacts_dir, cfg.model.backbone, Path(manifest.pairs_path))
    torch.save(
        {
            "backbone": cfg.model.backbone,
            "image_size": cfg.model.image_size,
            "pairs_path": manifest.pairs_path,
            "splits": splits,
        },
        output_path,
    )
    print(f"Saved features: {output_path.relative_to(REPO_DIR)} ({output_path.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
