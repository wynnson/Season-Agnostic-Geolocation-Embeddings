"""Train the projection head and season classifier on cached frozen-backbone features.

Same objective and checkpoint format as scripts/train.py with model.tuning: frozen.
Config values can be overridden from the command line, for example:
    uv run python scripts/train_cached.py training.season_loss_weight=0 checkpoint.run_name=no_grl
"""
import argparse
from pathlib import Path
import sys

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import pandas as pd
import torch

from omegaconf import OmegaConf
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from src.datasets.seasonet import SeasoNetDataset
from src.models.season_agnostic_model import load_model
from src.training.checkpoint import load_checkpoint, restore_training_state, save_checkpoint
from src.preprocessing.schema import Pairs
from src.training.evaluation import retrieval_recall_at_k
from src.training.feature_cache import (
    embed_features,
    feature_cache_path,
    forward_heads,
    load_feature_cache,
)
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
from src.utils.device import get_device


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Train heads on cached frozen-backbone features.")
    parser.add_argument("--resume", type=Path, help="Checkpoint path to resume from.")
    return parser.parse_known_args()


def build_hard_negative_index(train_pairs: pd.DataFrame, hard_negatives: pd.DataFrame, data_dir: Path, k: int) -> torch.Tensor:
    # Reuse the dataset's lookup so negatives match scripts/train.py exactly.
    dataset = SeasoNetDataset(
        train_pairs,
        split="train",
        data_dir=data_dir,
        hard_negatives=hard_negatives,
        sampled_hard_negatives_per_anchor=k,
    )
    row_by_loc_id = {loc_id: row for row, loc_id in enumerate(dataset.pairs[Pairs.LOC_ID])}
    return torch.tensor([
        [row_by_loc_id[negative] for negative in dataset.negative_ids_by_anchor[loc_id][:k]]
        for loc_id in dataset.pairs[Pairs.LOC_ID]
    ])


@torch.inference_mode()
def validate(model, summer: torch.Tensor, snow: torch.Tensor, chunk_size: int) -> ValidationMetrics:
    model.eval()
    summer_embeddings = embed_features(model, summer)
    snow_embeddings = embed_features(model, snow)
    return ValidationMetrics(
        summer_to_snow=retrieval_recall_at_k(summer_embeddings, snow_embeddings, chunk_size=chunk_size),
        snow_to_summer=retrieval_recall_at_k(snow_embeddings, summer_embeddings, chunk_size=chunk_size),
    )


def train_one_epoch(
    model,
    optimizer: torch.optim.Optimizer,
    criterion: InfoNCELoss,
    summer: torch.Tensor,
    snow: torch.Tensor,
    negative_index: torch.Tensor,
    batch_size: int,
    season_loss_weight: float,
    grl_scale: float,
    generator: torch.Generator,
) -> TrainingMetrics:
    model.train()
    device = summer.device
    order = torch.randperm(len(summer), generator=generator).to(device)
    num_batches = len(order) // batch_size  # drop_last, as in scripts/train.py

    totals = torch.zeros(6, device=device)
    progress = tqdm(range(num_batches), desc="Cached training", unit="batch", leave=False)
    for b in progress:
        idx = order[b * batch_size:(b + 1) * batch_size]
        negatives = negative_index[idx]

        summer_embeddings, summer_logits = forward_heads(model, summer[idx], grl_scale)
        snow_embeddings, snow_logits = forward_heads(model, snow[idx], grl_scale)
        summer_negative_embeddings = embed_features(model, summer[negatives])
        snow_negative_embeddings = embed_features(model, snow[negatives])

        contrastive_loss = (
            criterion(summer_embeddings, snow_embeddings, snow_negative_embeddings)
            + criterion(snow_embeddings, summer_embeddings, summer_negative_embeddings)
        ) / 2

        season_labels = torch.cat((
            torch.zeros(batch_size, dtype=torch.long, device=device),
            torch.ones(batch_size, dtype=torch.long, device=device),
        ))
        season_logits = torch.cat((summer_logits, snow_logits))
        season_loss = F.cross_entropy(season_logits, season_labels)
        season_predictions = season_logits.argmax(dim=1)
        season_classifier_accuracy = (season_predictions == season_labels).float().mean()
        summer_classifier_accuracy = (season_predictions[:batch_size] == 0).float().mean()
        snow_classifier_accuracy = (season_predictions[batch_size:] == 1).float().mean()
        total_loss = contrastive_loss + season_loss_weight * season_loss

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        totals += torch.stack((
            total_loss,
            contrastive_loss,
            season_loss,
            season_classifier_accuracy,
            summer_classifier_accuracy,
            snow_classifier_accuracy,
        )).detach()
        progress.set_postfix(
            contrastive=f"{contrastive_loss.item():.4f}",
            season=f"{season_loss.item():.4f}",
        )

    total, contrastive, season, classifier, summer, snow = (totals / num_batches).tolist()
    return TrainingMetrics(
        total_loss=total,
        contrastive_loss=contrastive,
        season_loss=season,
        season_classifier_accuracy=classifier,
        summer_classifier_accuracy=summer,
        snow_classifier_accuracy=snow,
    )


def main():
    args, overrides = parse_args()
    cfg = OmegaConf.merge(
        OmegaConf.load(REPO_DIR / "config/train_config.yaml"),
        OmegaConf.from_dotlist(overrides),
    )
    checkpoint_config = OmegaConf.to_container(cfg, resolve=True)
    if cfg.model.tuning != "frozen":
        raise ValueError("Cached features are only valid with model.tuning: frozen")

    torch.manual_seed(cfg.seed)
    generator = torch.Generator().manual_seed(cfg.seed)

    preprocessing_cfg = OmegaConf.load(REPO_DIR / "config/preprocessing_config.yaml")
    artifacts_dir = REPO_DIR / preprocessing_cfg.output.artifacts_dir
    manifest = OmegaConf.load(artifacts_dir / preprocessing_cfg.output.manifest_filename)
    data_dir = (REPO_DIR / cfg.data.data_dir).resolve()
    pairs = pd.read_parquet(REPO_DIR / manifest.pairs_path)
    hard_negatives = pd.read_parquet(REPO_DIR / manifest.hard_negatives_path)

    cache = load_feature_cache(
        feature_cache_path(artifacts_dir, cfg.model.backbone, Path(manifest.pairs_path)),
        pairs,
    )

    device = get_device()
    train_features = {s: cache["splits"]["train"][s].to(device) for s in ("summer", "snow")}
    val_features = {s: cache["splits"]["val"][s].to(device) for s in ("summer", "snow")}
    negative_index = build_hard_negative_index(
        pairs, hard_negatives, data_dir, cfg.training.sampled_hard_negatives_per_anchor
    ).to(device)

    # Full model (with the frozen backbone) so checkpoints load in evaluate.py and train.py.
    model = load_model(
        model_name=cfg.model.backbone,
        repository=cfg.model.repository,
        tuning=cfg.model.tuning,
        embedding_dim=cfg.model.projection_head.embedding_dim,
        hidden_dim=cfg.model.projection_head.hidden_dim,
        projection_dropout=cfg.model.projection_head.dropout,
        season_classes=cfg.model.season_classifier.num_classes,
        projection_architecture=cfg.model.projection_head.architecture,
        weights=cfg.model.weights,
    ).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = build_scheduler(
        optimizer,
        cfg.scheduler,
        cfg.training.epochs,
    )
    criterion = InfoNCELoss(cfg.training.temperature)

    checkpoint_prefix = REPO_DIR / cfg.checkpoint.output_dir / cfg.checkpoint.run_name
    writer = SummaryWriter(log_dir=REPO_DIR / cfg.logging.tensorboard_dir / cfg.checkpoint.run_name)
    writer.add_text(TRAINING_CONFIG_TEXT, OmegaConf.to_yaml(cfg.training, resolve=True), global_step=0,)

    print(f"Training {cfg.checkpoint.run_name} on cached features: "
          f"{len(train_features['summer']):,} train / {len(val_features['summer']):,} val pairs", flush=True)

    start_epoch, best_score, best_epoch = 1, float("-inf"), 0
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

    try:
        for epoch in range(start_epoch, cfg.training.epochs + 1):
            train_metrics = train_one_epoch(
                model, optimizer, criterion,
                train_features["summer"], train_features["snow"], negative_index,
                batch_size=cfg.training.batch_size,
                season_loss_weight=cfg.training.season_loss_weight,
                grl_scale=cfg.training.grl_scale,
                generator=generator,
            )
            validation_metrics = validate(
                model, val_features["summer"], val_features["snow"], cfg.training.retrieval_chunk_size
            )

            is_best = validation_metrics.mean_recall_at_1 > best_score
            if is_best:
                best_score, best_epoch = validation_metrics.mean_recall_at_1, epoch

            for tag, value in [
                (TRAIN_TOTAL_LOSS, train_metrics.total_loss),
                (TRAIN_CONTRASTIVE_HEAD_LOSS, train_metrics.contrastive_loss),
                (TRAIN_SEASON_CLASSIFIER_LOSS, train_metrics.season_loss),
                (TRAIN_SEASON_CLASSIFIER_ACCURACY, train_metrics.season_classifier_accuracy),
                (TRAIN_SUMMER_CLASSIFIER_ACCURACY, train_metrics.summer_classifier_accuracy),
                (TRAIN_SNOW_CLASSIFIER_ACCURACY, train_metrics.snow_classifier_accuracy),
                (TRAIN_LEARNING_RATE, optimizer.param_groups[0]["lr"]),
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
                f"S->W R@1={validation_metrics.summer_to_snow[RECALL_AT_1]:.4f} "
                f"W->S R@1={validation_metrics.snow_to_summer[RECALL_AT_1]:.4f} "
                f"mean_recall@1={validation_metrics.mean_recall_at_1:.4f}",
                flush=True,
            )

            scheduler.step()
            checkpoint_args = (
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
            if is_best:
                save_checkpoint(checkpoint_prefix.with_name(f"{checkpoint_prefix.name}_best.pt"), *checkpoint_args)
            save_checkpoint(checkpoint_prefix.with_name(f"{checkpoint_prefix.name}_last.pt"), *checkpoint_args)

        writer.add_hparams(
            hparam_dict=OmegaConf.to_container(cfg.training, resolve=True),
            metric_dict={HPARAM_BEST_MEAN_RECALL_AT_1: best_score},
            run_name=".",
        )
        print(f"Best epoch: {best_epoch} (mean_recall@1={best_score:.4f})", flush=True)
    finally:
        writer.close()


if __name__ == "__main__":
    main()
