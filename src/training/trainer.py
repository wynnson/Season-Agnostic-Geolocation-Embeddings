import math
import torch

from collections.abc import Callable
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.datasets.seasonet import SeasoNetData
from src.models.dinov2 import SeasonAgnosticDINOv2, preprocess_images
from src.training.evaluation import evaluate_cross_season_retrieval
from src.training.schema import TrainingHistory, TrainingMetrics, ValidationMetrics


class Trainer:
    def __init__(
        self,
        model: SeasonAgnosticDINOv2,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        image_size: int,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
        season_loss_weight: float,
        grl_scale: float,
        contrastive_criterion: torch.nn.Module,
        retrieval_chunk_size: int,
        log_every: int,
    ):
        self.model = model
        self.optimizer = optimizer
        self.contrastive_criterion = contrastive_criterion
        self.device = device
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.season_loss_weight = season_loss_weight
        self.grl_scale = grl_scale
        self.retrieval_chunk_size = retrieval_chunk_size
        self.log_every = log_every

    def prepare_images(self, images: torch.Tensor) -> torch.Tensor:
        return preprocess_images(
            images.to(self.device),
            image_size=self.image_size,
            mean=self.mean,
            std=self.std,
        )

    def embed_hard_negatives(self, images: torch.Tensor) -> torch.Tensor | None:
        batch_size, negatives_per_anchor, channels, height, width = images.shape
        if negatives_per_anchor == 0:
            return None

        images = images.reshape(
            batch_size * negatives_per_anchor,
            channels,
            height,
            width,
        )

        embeddings = self.model.forward_embeddings(self.prepare_images(images))
        return embeddings.reshape(batch_size, negatives_per_anchor, -1)

    def train_step(self, batch: SeasoNetData) -> TrainingMetrics:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        summer_images = self.prepare_images(batch.anchor)
        snow_images = self.prepare_images(batch.positive)

        summer_embeddings, summer_season_logits = self.model.forward_training(
            summer_images,
            grl_scale=self.grl_scale,
        )

        snow_embeddings, snow_season_logits = self.model.forward_training(
            snow_images,
            grl_scale=self.grl_scale,
        )

        summer_hard_negative_embeddings = self.embed_hard_negatives(
            batch.summer_hard_negatives
        )

        snow_hard_negative_embeddings = self.embed_hard_negatives(
            batch.snow_hard_negatives
        )

        summer_to_snow_loss = self.contrastive_criterion(
            query=summer_embeddings,
            key=snow_embeddings,
            hard_negative_keys=snow_hard_negative_embeddings,
        )

        snow_to_summer_loss = self.contrastive_criterion(
            query=snow_embeddings,
            key=summer_embeddings,
            hard_negative_keys=summer_hard_negative_embeddings,
        )

        contrastive_loss = (summer_to_snow_loss + snow_to_summer_loss) / 2

        season_logits = torch.cat((summer_season_logits, snow_season_logits))
        batch_size = len(summer_images)
        season_labels = torch.cat((
            torch.zeros(batch_size, dtype=torch.long, device=self.device),
            torch.ones(batch_size, dtype=torch.long, device=self.device),
        ))
        season_loss = F.cross_entropy(season_logits, season_labels)

        total_loss = contrastive_loss + self.season_loss_weight * season_loss
        total_loss.backward()
        self.optimizer.step()

        return TrainingMetrics(
            total_loss=total_loss.item(),
            contrastive_loss=contrastive_loss.item(),
            season_loss=season_loss.item(),
        )

    def train_one_epoch(self, data_loader: DataLoader) -> TrainingMetrics:
        self.model.train()

        total_loss = 0.0
        contrastive_loss = 0.0
        season_loss = 0.0
        total_samples = 0

        for batch in tqdm(data_loader, desc="Training", leave=False):
            metrics = self.train_step(batch)
            batch_size = batch.anchor.size(0)

            total_loss += metrics.total_loss * batch_size
            contrastive_loss += metrics.contrastive_loss * batch_size
            season_loss += metrics.season_loss * batch_size
            total_samples += batch_size

        return TrainingMetrics(
            total_loss=total_loss / total_samples,
            contrastive_loss=contrastive_loss / total_samples,
            season_loss=season_loss / total_samples,
        )

    def validate(self, data_loader: DataLoader) -> ValidationMetrics:
        self.model.eval()

        def embed_images(images: torch.Tensor) -> torch.Tensor:
            return self.model.forward_embeddings(self.prepare_images(images))

        return evaluate_cross_season_retrieval(
            data_loader=data_loader,
            embed_fn=embed_images,
            chunk_size=self.retrieval_chunk_size,
            log_every=self.log_every,
        )

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        start_epoch: int = 1,
        best_score: float = -math.inf,
        best_epoch: int = 0,
        on_epoch_end: Callable[
            [int, TrainingMetrics, ValidationMetrics, bool, float, int],
            None,
        ] | None = None,
    ) -> TrainingHistory:
        train_hist = []
        val_hist = []

        for epoch in range(start_epoch, epochs + 1):
            train_metrics = self.train_one_epoch(train_loader)
            val_metrics = self.validate(val_loader)

            train_hist.append(train_metrics)
            val_hist.append(val_metrics)

            current_score = val_metrics.mean_recall_at_1
            is_best = current_score > best_score
            if is_best:
                best_score = current_score
                best_epoch = epoch

            if on_epoch_end is not None:
                on_epoch_end(
                    epoch,
                    train_metrics,
                    val_metrics,
                    is_best,
                    best_score,
                    best_epoch,
                )

        return TrainingHistory(
            train=train_hist,
            validation=val_hist,
            best_epoch=best_epoch,
            best_score=best_score,
        )
