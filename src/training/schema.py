from dataclasses import asdict, dataclass
from typing import Any


RECALL_AT_1                                         = "recall@1"
RECALL_AT_5                                         = "recall@5"

TRAIN_TOTAL_LOSS                                    = "train/total_loss"
TRAIN_CONTRASTIVE_HEAD_LOSS                         = "train/contrastive_head/loss"
TRAIN_SEASON_CLASSIFIER_LOSS                        = "train/season_classifier/loss"
TRAIN_SEASON_CLASSIFIER_ACCURACY                    = "train/season_classifier/accuracy"
TRAIN_SUMMER_CLASSIFIER_ACCURACY                    = "train/season_classifier/summer_accuracy"
TRAIN_SNOW_CLASSIFIER_ACCURACY                      = "train/season_classifier/snow_accuracy"
TRAIN_LEARNING_RATE                                 = "train/learning_rate"

VALIDATION_RETRIEVAL_MEAN_RECALL_AT_1               = "validation/retrieval/mean_recall_at_1"
VALIDATION_RETRIEVAL_SUMMER_TO_SNOW_RECALL_AT_1     = "validation/retrieval/summer_to_snow_recall_at_1"
VALIDATION_RETRIEVAL_SNOW_TO_SUMMER_RECALL_AT_1     = "validation/retrieval/snow_to_summer_recall_at_1"
VALIDATION_RETRIEVAL_SUMMER_TO_SNOW_RECALL_AT_5     = "validation/retrieval/summer_to_snow_recall_at_5"
VALIDATION_RETRIEVAL_SNOW_TO_SUMMER_RECALL_AT_5     = "validation/retrieval/snow_to_summer_recall_at_5"

TRAINING_CONFIG_TEXT                                = "config/training"
HPARAM_BEST_MEAN_RECALL_AT_1                        = "hparams/best_mean_recall_at_1"


@dataclass
class TrainingMetrics:
    total_loss: float
    contrastive_loss: float
    season_loss: float
    attention_consistency_loss: float = 0.0
    season_classifier_accuracy: float = float("nan")
    summer_classifier_accuracy: float = float("nan")
    snow_classifier_accuracy: float = float("nan")


@dataclass
class ValidationMetrics:
    summer_to_snow: dict[str, float]
    snow_to_summer: dict[str, float]

    @property
    def mean_recall_at_1(self) -> float:
        return (
            self.summer_to_snow[RECALL_AT_1]
            + self.snow_to_summer[RECALL_AT_1]
        ) / 2


@dataclass
class TrainingHistory:
    train: list[TrainingMetrics]
    validation: list[ValidationMetrics]
    best_epoch: int
    best_score: float


@dataclass
class TrainingCheckpoint:
    epoch: int
    model_state_dict: dict[str, Any]
    optimizer_state_dict: dict[str, Any]
    scheduler_state_dict: dict[str, Any] | None
    train_metrics: TrainingMetrics
    validation_metrics: ValidationMetrics
    best_score: float
    best_epoch: int
    config: dict[str, Any]

    @classmethod
    def from_dict(cls, checkpoint: dict[str, Any]) -> "TrainingCheckpoint":
        return cls(
            epoch=checkpoint["epoch"],
            model_state_dict=checkpoint["model_state_dict"],
            optimizer_state_dict=checkpoint["optimizer_state_dict"],
            scheduler_state_dict=checkpoint.get("scheduler_state_dict"),
            train_metrics=TrainingMetrics(**checkpoint["train_metrics"]),
            validation_metrics=ValidationMetrics(**checkpoint["validation_metrics"]),
            best_score=checkpoint["best_score"],
            best_epoch=checkpoint["best_epoch"],
            config=checkpoint["config"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "model_state_dict": self.model_state_dict,
            "optimizer_state_dict": self.optimizer_state_dict,
            "scheduler_state_dict": self.scheduler_state_dict,
            "train_metrics": asdict(self.train_metrics),
            "validation_metrics": asdict(self.validation_metrics),
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "config": self.config,
        }
