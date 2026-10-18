from pathlib import Path
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from preprocessing.data_cleaner import load_pairs, attach_splits
from src.datasets.seasonet import SeasoNetDataset
# from training.trainer import Trainer


def main():
    TRAIN_CFG_PATH = "/config/train_config.yaml"
    cfg = OmegaConf.load(TRAIN_CFG_PATH)
    
    data_dir = Path(cfg.data.data_dir)
    pairs = load_pairs(data_dir)
    pairs = attach_splits(pairs, data_dir)

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
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
    )

    # TODO: load model stuff & init model
    # model = model()

if __name__ == "__main__":
    main()
