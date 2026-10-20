from pathlib import Path
from typing import Any

import torch

from src.training.schema import TrainingCheckpoint, TrainingMetrics, ValidationMetrics


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    train_metrics: TrainingMetrics,
    validation_metrics: ValidationMetrics,
    best_score: float,
    best_epoch: int,
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = TrainingCheckpoint(
        epoch=epoch,
        model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        scheduler_state_dict=scheduler.state_dict(),
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        best_score=best_score,
        best_epoch=best_epoch,
        config=config,
    )
    torch.save(checkpoint.to_dict(), path)


def load_checkpoint(path: Path, device: torch.device) -> TrainingCheckpoint:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    return TrainingCheckpoint.from_dict(checkpoint)


def restore_training_state(
    checkpoint: TrainingCheckpoint,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> None:
    model.load_state_dict(checkpoint.model_state_dict)
    optimizer.load_state_dict(checkpoint.optimizer_state_dict)

    if checkpoint.scheduler_state_dict is None:
        scheduler.step(checkpoint.epoch)
    else:
        scheduler.load_state_dict(checkpoint.scheduler_state_dict)
