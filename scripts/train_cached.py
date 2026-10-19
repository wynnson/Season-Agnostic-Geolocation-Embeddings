"""Train the projection head and season classifier on cached frozen-DINOv2 features.

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

from scripts.train import (
    TRAIN_CONTRASTIVE_LOSS,
    TRAIN_SEASON_LOSS,
    TRAIN_TOTAL_LOSS,
    VALIDATION_MEAN_RECALL_AT_1,
    VALIDATION_SNOW_TO_SUMMER_RECALL_AT_1,
    VALIDATION_SNOW_TO_SUMMER_RECALL_AT_5,
    VALIDATION_SUMMER_TO_SNOW_RECALL_AT_1,
    VALIDATION_SUMMER_TO_SNOW_RECALL_AT_5,
    save_checkpoint,
)
from src.datasets.seasonet import SeasoNetDataset
from src.models.dinov2 import load_dinov2
from src.preprocessing.schema import Pairs
from src.training.evaluation import retrieval_recall_at_k
from src.training.feature_cache import (
    embed_features,
    feature_cache_path,
    forward_heads,
    load_feature_cache,
)
from src.training.losses.info_nce import InfoNCELoss
from src.training.schema import RECALL_AT_1, RECALL_AT_5, TrainingMetrics, ValidationMetrics
from src.utils.device import get_device


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Train heads on cached frozen DINOv2 features.")
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

    totals = torch.zeros(3, device=device)
    for b in range(num_batches):
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
        season_loss = F.cross_entropy(torch.cat((summer_logits, snow_logits)), season_labels)
        total_loss = contrastive_loss + season_loss_weight * season_loss

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        totals += torch.stack((total_loss, contrastive_loss, season_loss)).detach()

    total, contrastive, season = (totals / num_batches).tolist()
    return TrainingMetrics(total_loss=total, contrastive_loss=contrastive, season_loss=season)


def main():
    _, overrides = parse_args()
    cfg = OmegaConf.merge(
        OmegaConf.load(REPO_DIR / "config/train_config.yaml"),
        OmegaConf.from_dotlist(overrides),
    )
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
    model = load_dinov2(
        model_name=cfg.model.backbone,
        repository=cfg.model.repository,
        tuning=cfg.model.tuning,
        embedding_dim=cfg.model.projection_head.embedding_dim,
        hidden_dim=cfg.model.projection_head.hidden_dim,
        projection_dropout=cfg.model.projection_head.dropout,
        season_classes=cfg.model.season_classifier.num_classes,
    ).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    criterion = InfoNCELoss(cfg.training.temperature)

    checkpoint_prefix = REPO_DIR / cfg.checkpoint.output_dir / cfg.checkpoint.run_name
    writer = SummaryWriter(log_dir=REPO_DIR / cfg.logging.tensorboard_dir / cfg.checkpoint.run_name)
    print(f"Training {cfg.checkpoint.run_name} on cached features: "
          f"{len(train_features['summer']):,} train / {len(val_features['summer']):,} val pairs", flush=True)

    best_score, best_epoch = float("-inf"), 0
    try:
        for epoch in range(1, cfg.training.epochs + 1):
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
                (TRAIN_CONTRASTIVE_LOSS, train_metrics.contrastive_loss),
                (TRAIN_SEASON_LOSS, train_metrics.season_loss),
                (VALIDATION_MEAN_RECALL_AT_1, validation_metrics.mean_recall_at_1),
                (VALIDATION_SUMMER_TO_SNOW_RECALL_AT_1, validation_metrics.summer_to_snow[RECALL_AT_1]),
                (VALIDATION_SNOW_TO_SUMMER_RECALL_AT_1, validation_metrics.snow_to_summer[RECALL_AT_1]),
                (VALIDATION_SUMMER_TO_SNOW_RECALL_AT_5, validation_metrics.summer_to_snow[RECALL_AT_5]),
                (VALIDATION_SNOW_TO_SUMMER_RECALL_AT_5, validation_metrics.snow_to_summer[RECALL_AT_5]),
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

            checkpoint_args = (model, optimizer, epoch, train_metrics, validation_metrics, best_score, best_epoch, cfg)
            if epoch % cfg.checkpoint.save_every == 0:
                save_checkpoint(checkpoint_prefix.with_name(f"{checkpoint_prefix.name}_epoch_{epoch:03d}.pt"), *checkpoint_args)
            if is_best:
                save_checkpoint(checkpoint_prefix.with_name(f"{checkpoint_prefix.name}_best.pt"), *checkpoint_args)
            save_checkpoint(checkpoint_prefix.with_name(f"{checkpoint_prefix.name}_last.pt"), *checkpoint_args)

        print(f"Best epoch: {best_epoch} (mean_recall@1={best_score:.4f})", flush=True)
    finally:
        writer.close()


if __name__ == "__main__":
    main()
