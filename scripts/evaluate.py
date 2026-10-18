from pathlib import Path
import sys

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import pandas as pd
import torch

from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.datasets.seasonet import SeasoNetDataset
from src.models.dinov2 import embed_dinov2_images, load_frozen_dinov2
from src.training.evaluation import evaluate_cross_season_retrieval
from src.utils.device import get_device


def main():
    cfg = OmegaConf.load(REPO_DIR / "config/evaluation_config.yaml")
    preprocessing_cfg = OmegaConf.load(REPO_DIR / "config/preprocessing_config.yaml")
    manifest_path = (
        REPO_DIR
        / preprocessing_cfg.output.artifacts_dir
        / preprocessing_cfg.output.manifest_filename
    )
    artifact_manifest = OmegaConf.load(manifest_path)
    torch.manual_seed(cfg.seed)

    data_dir = (REPO_DIR / cfg.data.data_dir).resolve()
    pairs = pd.read_parquet(REPO_DIR / artifact_manifest.pairs_path)

    val_dataset = SeasoNetDataset(pairs, split="val", data_dir=data_dir)
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.evaluation.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
    )

    device = get_device()

    if cfg.evaluation.trained:
        raise NotImplementedError("Trained checkpoint evaluation is not implemented yet")

    model = load_frozen_dinov2(
        cfg.model.backbone,
        cfg.model.repository,
    ).to(device)
    
    model.eval()

    def embed_fn(images: torch.Tensor) -> torch.Tensor:
        return embed_dinov2_images(
            model,
            images.to(device),
            cfg.model.image_size,
            tuple(cfg.model.normalization.mean),
            tuple(cfg.model.normalization.std),
        )

    metrics = evaluate_cross_season_retrieval(
        val_loader,
        embed_fn=embed_fn,
        chunk_size=cfg.evaluation.retrieval_chunk_size,
        log_every=cfg.logging.log_every,
    )

    print(f"DINOv2 evaluation on geographic validation split ({len(val_dataset):,} pairs)")
    print("Summer -> Snow:", metrics.summer_to_snow)
    print("Snow -> Summer:", metrics.snow_to_summer)


if __name__ == "__main__":
    main()
