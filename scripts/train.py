import argparse
from pathlib import Path
import sys
import pandas as pd
import torch

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.datasets.seasonet import SeasoNetDataset
from src.models.dinov2 import backbone_weights_path, load_dinov2
from src.training.checkpoint import load_checkpoint, restore_training_state, save_checkpoint
from src.training.losses.info_nce import InfoNCELoss
from src.training.scheduler import build_scheduler
from src.training.schema import (
    RECALL_AT_1,
    RECALL_AT_5,
    HPARAM_BEST_MEAN_RECALL_AT_1,
    TRAIN_CONTRASTIVE_HEAD_LOSS,
    TRAIN_LEARNING_RATE,
    TRAIN_SEASON_CLASSIFIER_ACCURACY,
    TRAIN_SEASON_CLASSIFIER_LOSS,
    TRAIN_SNOW_CLASSIFIER_ACCURACY,
    TRAIN_SUMMER_CLASSIFIER_ACCURACY,
    TRAIN_TOTAL_LOSS,
    TRAINING_CONFIG_TEXT,
    TrainingMetrics,
    ValidationMetrics,
    VALIDATION_RETRIEVAL_MEAN_RECALL_AT_1,
    VALIDATION_RETRIEVAL_SNOW_TO_SUMMER_RECALL_AT_1,
    VALIDATION_RETRIEVAL_SNOW_TO_SUMMER_RECALL_AT_5,
    VALIDATION_RETRIEVAL_SUMMER_TO_SNOW_RECALL_AT_1,
    VALIDATION_RETRIEVAL_SUMMER_TO_SNOW_RECALL_AT_5,
)
from src.training.trainer import Trainer
from src.utils.device import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train season-agnostic DINOv2 embeddings.")
    parser.add_argument(
        "--resume",
        type=Path,
        help="Last checkpoint path from an interrupted run.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    train_cfg_path = REPO_DIR / "config/train_config.yaml"
    cfg = OmegaConf.load(train_cfg_path)
    checkpoint_config = OmegaConf.to_container(cfg, resolve=True)
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
        projection_architecture=cfg.model.projection_head.architecture,
        weights=backbone_weights_path(cfg.model, REPO_DIR),
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

    scheduler = build_scheduler(
        optimizer,
        cfg.scheduler,
        cfg.training.epochs,
    )

    start_epoch = 1
    best_score = float("-inf")
    best_epoch = 0
    if args.resume is not None:
        checkpoint_path = args.resume.expanduser().resolve()
        checkpoint = load_checkpoint(checkpoint_path, device)
        restore_training_state(checkpoint, model, optimizer, scheduler)
        start_epoch = checkpoint.epoch + 1
        best_score = checkpoint.best_score
        best_epoch = checkpoint.best_epoch

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

    writer = SummaryWriter(log_dir=REPO_DIR / cfg.logging.tensorboard_dir)
    writer.add_text(TRAINING_CONFIG_TEXT, OmegaConf.to_yaml(cfg.training, resolve=True), global_step=0,)

    def on_epoch_end(
        epoch: int,
        train_metrics: TrainingMetrics,
        validation_metrics: ValidationMetrics,
        is_best: bool,
        best_score: float,
        best_epoch: int,
    ) -> None:
        for tag, value in [
            (TRAIN_TOTAL_LOSS, train_metrics.total_loss),
            (TRAIN_CONTRASTIVE_HEAD_LOSS, train_metrics.contrastive_loss),
            (TRAIN_SEASON_CLASSIFIER_LOSS, train_metrics.season_loss),
            (TRAIN_SEASON_CLASSIFIER_ACCURACY, train_metrics.season_classifier_accuracy),
            (TRAIN_SUMMER_CLASSIFIER_ACCURACY, train_metrics.summer_classifier_accuracy),
            (TRAIN_SNOW_CLASSIFIER_ACCURACY, train_metrics.snow_classifier_accuracy),
            (TRAIN_LEARNING_RATE, optimizer.param_groups[0]["lr"]),
        ]:
            writer.add_scalar(tag, value, epoch)
        for tag, value in [
            (VALIDATION_RETRIEVAL_MEAN_RECALL_AT_1, validation_metrics.mean_recall_at_1),
            (VALIDATION_RETRIEVAL_SUMMER_TO_SNOW_RECALL_AT_1, validation_metrics.summer_to_snow[RECALL_AT_1]),
            (VALIDATION_RETRIEVAL_SNOW_TO_SUMMER_RECALL_AT_1, validation_metrics.snow_to_summer[RECALL_AT_1]),
            (VALIDATION_RETRIEVAL_SUMMER_TO_SNOW_RECALL_AT_5, validation_metrics.summer_to_snow[RECALL_AT_5]),
            (VALIDATION_RETRIEVAL_SNOW_TO_SUMMER_RECALL_AT_5, validation_metrics.snow_to_summer[RECALL_AT_5]),
        ]:
            writer.add_scalar(tag, value, epoch)

        print(
            f"Epoch {epoch}/{cfg.training.epochs} "
            f"loss={train_metrics.total_loss:.4f} "
            f"contrastive={train_metrics.contrastive_loss:.4f} "
            f"season={train_metrics.season_loss:.4f} "
            f"mean_recall@1={validation_metrics.mean_recall_at_1:.4f}",
            flush=True,
        )

        scheduler.step()
        if is_best:
            save_checkpoint(
                checkpoint_prefix.with_name(f"{checkpoint_prefix.name}_best.pt"),
                model,
                optimizer,
                scheduler,
                epoch,
                train_metrics,
                validation_metrics,
                best_score,
                best_epoch,
                checkpoint_config,
            )
        save_checkpoint(
            checkpoint_prefix.with_name(f"{checkpoint_prefix.name}_last.pt"),
            model,
            optimizer,
            scheduler,
            epoch,
            train_metrics,
            validation_metrics,
            best_score,
            best_epoch,
            checkpoint_config,
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

        writer.add_hparams(
            hparam_dict=OmegaConf.to_container(cfg.training, resolve=True),
            metric_dict={HPARAM_BEST_MEAN_RECALL_AT_1: history.best_score},
            run_name=".",
        )
        print(f"Best epoch: {history.best_epoch}", flush=True)
    finally:
        writer.close()

if __name__ == "__main__":
    main()
