from dataclasses import dataclass


RECALL_AT_1 = "recall@1"
RECALL_AT_5 = "recall@5"


@dataclass
class TrainingMetrics:
    total_loss: float
    contrastive_loss: float
    season_loss: float


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
