import argparse
from pathlib import Path
import sys
import pandas as pd
import torch

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from dataclasses import asdict
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.datasets.seasonet import SeasoNetDataset
from src.models.dinov2 import load_dinov2
from src.training.losses.info_nce import InfoNCELoss
from src.training.schema import (
    RECALL_AT_1,
    RECALL_AT_5,
    TrainingMetrics,
    ValidationMetrics,
)
from src.training.trainer import Trainer
from src.utils.device import get_device


TRAIN_TOTAL_LOSS = "train/total_loss"
TRAIN_CONTRASTIVE_LOSS = "train/contrastive_loss"
TRAIN_SEASON_LOSS = "train/season_loss"
VALIDATION_MEAN_RECALL_AT_1 = "validation/mean_recall_at_1"
VALIDATION_SUMMER_TO_SNOW_RECALL_AT_1 = "validation/summer_to_snow_recall_at_1"
VALIDATION_SNOW_TO_SUMMER_RECALL_AT_1 = "validation/snow_to_summer_recall_at_1"
VALIDATION_SUMMER_TO_SNOW_RECALL_AT_5 = "validation/summer_to_snow_recall_at_5"
VALIDATION_SNOW_TO_SUMMER_RECALL_AT_5 = "validation/snow_to_summer_recall_at_5"


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_metrics: TrainingMetrics,
    validation_metrics: ValidationMetrics,
    best_score: float,
    best_epoch: int,
    cfg,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_metrics": asdict(train_metrics),
            "validation_metrics": asdict(validation_metrics),
            "best_score": best_score,
            "best_epoch": best_epoch,
            "config": OmegaConf.to_container(cfg, resolve=True),
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train season-agnostic DINOv2 embeddings.")
    parser.add_argument(
        "--resume",
        type=Path,
        help="Checkpoint path to resume from.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    train_cfg_path = REPO_DIR / "config/train_config.yaml"
    cfg = OmegaConf.load(train_cfg_path)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    preprocessing_cfg = OmegaConf.load(REPO_DIR / "config/preprocessing_config.yaml")
    manifest_path = (
        REPO_DIR
        / preprocessing_cfg.output.artifacts_dir
        / preprocessing_cfg.output.manifest_filename
    )
    artifact_manifest = OmegaConf.load(manifest_path)

    data_dir = (REPO_DIR / cfg.data.data_dir).resolve()
    pairs = pd.read_parquet(REPO_DIR / artifact_manifest.pairs_path)
    hard_negatives = pd.read_parquet(REPO_DIR / artifact_manifest.hard_negatives_path)

    train_dataset = SeasoNetDataset(
        pairs=pairs,
        split="train",
        data_dir=data_dir,
        transform=None,
        hard_negatives=hard_negatives,
        sampled_hard_negatives_per_anchor=(
            cfg.training.sampled_hard_negatives_per_anchor
        ),
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
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.data.num_workers > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.data.num_workers > 0,
    )

    device = get_device()
    model = load_dinov2(
        model_name=cfg.model.backbone,
        repository=cfg.model.repository,
        tuning=cfg.model.tuning,
        embedding_dim=cfg.model.projection_head.embedding_dim,
        hidden_dim=cfg.model.projection_head.hidden_dim,
        projection_dropout=cfg.model.projection_head.dropout,
        season_classes=cfg.model.season_classifier.num_classes,
    ).to(device)

    if cfg.optimizer.name != "adamw":
        raise ValueError(f"Unsupported optimizer: {cfg.optimizer.name}")

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    start_epoch = 1
    best_score = float("-inf")
    best_epoch = 0
    if args.resume is not None:
        checkpoint_path = args.resume.expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1

        if "best_score" in checkpoint:
            best_score = float(checkpoint["best_score"])
            best_epoch = int(checkpoint["best_epoch"])
        elif "validation_metrics" in checkpoint:
            validation_metrics = ValidationMetrics(**checkpoint["validation_metrics"])
            best_score = validation_metrics.mean_recall_at_1
            best_epoch = checkpoint["epoch"]

        if start_epoch > cfg.training.epochs:
            raise ValueError(
                "Checkpoint already reached the configured number of epochs; "
                "increase training.epochs to continue."
            )
        print(f"Resuming from epoch {start_epoch}", flush=True)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        image_size=cfg.model.image_size,
        mean=tuple(cfg.model.normalization.mean),
        std=tuple(cfg.model.normalization.std),
        season_loss_weight=cfg.training.season_loss_weight,
        grl_scale=cfg.training.grl_scale,
        contrastive_criterion=InfoNCELoss(cfg.training.temperature),
        retrieval_chunk_size=cfg.training.retrieval_chunk_size,
        log_every=cfg.logging.log_every,
    )

    checkpoint_dir = REPO_DIR / cfg.checkpoint.output_dir
    checkpoint_prefix = checkpoint_dir / cfg.checkpoint.run_name
    if cfg.checkpoint.save_every < 1:
        raise ValueError("checkpoint.save_every must be at least 1")

    writer = SummaryWriter(log_dir=REPO_DIR / cfg.logging.tensorboard_dir)

    def on_epoch_end(
        epoch: int,
        train_metrics: TrainingMetrics,
        validation_metrics: ValidationMetrics,
        is_best: bool,
        best_score: float,
        best_epoch: int,
    ) -> None:
        writer.add_scalar(TRAIN_TOTAL_LOSS, train_metrics.total_loss, epoch)
        writer.add_scalar(TRAIN_CONTRASTIVE_LOSS, train_metrics.contrastive_loss, epoch)
        writer.add_scalar(TRAIN_SEASON_LOSS, train_metrics.season_loss, epoch)
        writer.add_scalar(
            VALIDATION_MEAN_RECALL_AT_1,
            validation_metrics.mean_recall_at_1,
            epoch,
        )
        writer.add_scalar(
            VALIDATION_SUMMER_TO_SNOW_RECALL_AT_1,
            validation_metrics.summer_to_snow[RECALL_AT_1],
            epoch,
        )
        writer.add_scalar(
            VALIDATION_SNOW_TO_SUMMER_RECALL_AT_1,
            validation_metrics.snow_to_summer[RECALL_AT_1],
            epoch,
        )
        writer.add_scalar(
            VALIDATION_SUMMER_TO_SNOW_RECALL_AT_5,
            validation_metrics.summer_to_snow[RECALL_AT_5],
            epoch,
        )
        writer.add_scalar(
            VALIDATION_SNOW_TO_SUMMER_RECALL_AT_5,
            validation_metrics.snow_to_summer[RECALL_AT_5],
            epoch,
        )

        print(
            f"Epoch {epoch}/{cfg.training.epochs} "
            f"loss={train_metrics.total_loss:.4f} "
            f"contrastive={train_metrics.contrastive_loss:.4f} "
            f"season={train_metrics.season_loss:.4f} "
            f"mean_recall@1={validation_metrics.mean_recall_at_1:.4f}",
            flush=True,
        )

        if epoch % cfg.checkpoint.save_every == 0:
            save_checkpoint(
                checkpoint_prefix.with_name(
                    f"{checkpoint_prefix.name}_epoch_{epoch:03d}.pt"
                ),
                model,
                optimizer,
                epoch,
                train_metrics,
                validation_metrics,
                best_score,
                best_epoch,
                cfg,
            )
        if is_best:
            save_checkpoint(
                checkpoint_prefix.with_name(f"{checkpoint_prefix.name}_best.pt"),
                model,
                optimizer,
                epoch,
                train_metrics,
                validation_metrics,
                best_score,
                best_epoch,
                cfg,
            )
        save_checkpoint(
            checkpoint_prefix.with_name(f"{checkpoint_prefix.name}_last.pt"),
            model,
            optimizer,
            epoch,
            train_metrics,
            validation_metrics,
            best_score,
            best_epoch,
            cfg,
        )

    try:
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=cfg.training.epochs,
            start_epoch=start_epoch,
            best_score=best_score,
            best_epoch=best_epoch,
            on_epoch_end=on_epoch_end,
        )

        print(f"Best epoch: {history.best_epoch}", flush=True)
    finally:
        writer.close()

if __name__ == "__main__":
    main()
