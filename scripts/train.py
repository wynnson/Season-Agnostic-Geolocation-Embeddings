from pathlib import Path
import sys

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import pandas as pd

from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.datasets.seasonet import SeasoNetDataset


def main():
    train_cfg_path = REPO_DIR / "config/train_config.yaml"
    cfg = OmegaConf.load(train_cfg_path)

    data_dir = (REPO_DIR / cfg.data.data_dir).resolve()
    pairs = pd.read_parquet(REPO_DIR / cfg.data.pairs_path)

    train_dataset = SeasoNetDataset(
        pairs=pairs,
        split="train",
        data_dir=data_dir,
        transform=None,
    )

    val_dataset = SeasoNetDataset(
        pairs=pairs,
        split="val",
        data_dir=data_dir,
        transform=None,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
    )

    # TODO: initialize a trainable model, optimizer, loss, and trainer.

if __name__ == "__main__":
    main()
