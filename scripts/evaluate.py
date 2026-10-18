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
from src.models.dinov2 import embed_dinov2_images, load_dinov2
from src.training.evaluation import evaluate_cross_season_retrieval
from src.training.schema import Metrics
from src.utils.device import get_device


def main():
    cfg = OmegaConf.load(REPO_DIR / "config/evaluation_config.yaml")
    torch.manual_seed(cfg.seed)

    data_dir = (REPO_DIR / cfg.data.data_dir).resolve()
    pairs = pd.read_parquet(REPO_DIR / cfg.data.pairs_path)

    val_dataset = SeasoNetDataset(pairs, split="val", data_dir=data_dir)
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.evaluation.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
    )

    device = get_device()

    model = load_dinov2(
        cfg.model.backbone,
        trained=cfg.evaluation.trained,
        repository=cfg.model.repository,
    ).to(device)
    
    model.eval()

    def embed_fn(images: torch.Tensor) -> torch.Tensor:
        return embed_dinov2_images(model, images.to(device), cfg.model.image_size)

    metrics = evaluate_cross_season_retrieval(
        val_loader,
        embed_fn=embed_fn,
        chunk_size=cfg.evaluation.retrieval_chunk_size,
        log_every=cfg.logging.log_every,
    )

    print(f"DINOv2 evaluation on geographic validation split ({len(val_dataset):,} pairs)")
    print("Summer -> Snow:", metrics[Metrics.SUMMER_TO_SNOW])
    print("Snow -> Summer:", metrics[Metrics.SNOW_TO_SUMMER])


if __name__ == "__main__":
    main()
