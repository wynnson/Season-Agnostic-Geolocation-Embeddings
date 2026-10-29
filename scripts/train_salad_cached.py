"""Train a frozen-DINOv3 SALAD descriptor from cached CLS and patch features."""
import argparse
from pathlib import Path
import sys

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import pandas as pd
import torch

from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from scripts.train_cached import batch_features, build_hard_negative_index
from src.models.salad import SALADPool
from src.training.checkpoint import load_checkpoint, restore_training_state, save_checkpoint
from src.training.evaluation import retrieval_recall_at_k
from src.training.feature_cache import load_patch_feature_cache, patch_feature_cache_path
from src.training.losses.info_nce import InfoNCELoss
from src.training.scheduler import build_scheduler
from src.training.schema import (
    RECALL_AT_1,
    RECALL_AT_5,
    TRAIN_CONTRASTIVE_HEAD_LOSS,
    TRAIN_LEARNING_RATE,
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
    parser = argparse.ArgumentParser(description="Train SALAD from cached DINO CLS and patch features.")
    parser.add_argument("--resume", type=Path, help="Checkpoint path to resume from.")
    return parser.parse_known_args()


def embed(model: SALADPool, features: dict[str, torch.Tensor], indices: torch.Tensor, device: torch.device) -> torch.Tensor:
    cls_features, patch_features = batch_features(features, indices, device)
    if patch_features is None:
        raise ValueError("SALAD requires cached patch features")
    return model(cls_features, patch_features)


@torch.inference_mode()
def validate(model: SALADPool, summer: dict, snow: dict, chunk_size: int) -> ValidationMetrics:
    model.eval()
    device = next(model.parameters()).device

    def embed_all(features: dict) -> torch.Tensor:
        return torch.cat([
            embed(model, features, torch.arange(start, min(start + chunk_size, len(features["cls"]))), device)
            for start in range(0, len(features["cls"]), chunk_size)
        ])

    summer_embeddings = embed_all(summer)
    snow_embeddings = embed_all(snow)
    return ValidationMetrics(
        summer_to_snow=retrieval_recall_at_k(summer_embeddings, snow_embeddings, chunk_size=chunk_size),
        snow_to_summer=retrieval_recall_at_k(snow_embeddings, summer_embeddings, chunk_size=chunk_size),
    )


def train_one_epoch(
    model: SALADPool,
    optimizer: torch.optim.Optimizer,
    criterion: InfoNCELoss,
    summer: dict,
    snow: dict,
    negative_index: torch.Tensor | None,
    batch_size: int,
    generator: torch.Generator,
) -> TrainingMetrics:
    model.train()
    device = next(model.parameters()).device
    order = torch.randperm(len(summer["cls"]), generator=generator)
    num_batches = len(order) // batch_size
    totals = torch.zeros(2, device=device)
    progress = tqdm(range(num_batches), desc="Cached SALAD training", unit="batch", leave=False)

    for batch_number in progress:
        indices = order[batch_number * batch_size:(batch_number + 1) * batch_size]
        summer_embeddings = embed(model, summer, indices, device)
        snow_embeddings = embed(model, snow, indices, device)
        if negative_index is None:
            summer_negatives = snow_negatives = None
        else:
            negatives = negative_index[indices]
            summer_negatives = embed(model, summer, negatives.flatten(), device).reshape(batch_size, -1, model.output_dim)
            snow_negatives = embed(model, snow, negatives.flatten(), device).reshape(batch_size, -1, model.output_dim)

        contrastive_loss = (
            criterion(summer_embeddings, snow_embeddings, snow_negatives)
            + criterion(snow_embeddings, summer_embeddings, summer_negatives)
        ) / 2
        optimizer.zero_grad(set_to_none=True)
        contrastive_loss.backward()
        optimizer.step()
        totals += torch.stack((contrastive_loss, contrastive_loss.detach()))
        progress.set_postfix(contrastive=f"{contrastive_loss.item():.4f}")

    total_loss, contrastive_loss = (totals / num_batches).tolist()
    return TrainingMetrics(total_loss=total_loss, contrastive_loss=contrastive_loss, season_loss=0.0)


def main() -> None:
    args, overrides = parse_args()
    cfg = OmegaConf.merge(
        OmegaConf.load(REPO_DIR / "config/train_config.yaml"),
        OmegaConf.from_dotlist(overrides),
    )
    if cfg.model.tuning != "frozen":
        raise ValueError("Cached SALAD training is only valid with model.tuning: frozen")
    salad_cfg = cfg.model.salad
    checkpoint_config = OmegaConf.to_container(cfg, resolve=True)
    if salad_cfg.num_clusters * salad_cfg.cluster_dim + salad_cfg.token_dim != cfg.model.projection_head.embedding_dim:
        raise ValueError("SALAD descriptor dimension must equal model.projection_head.embedding_dim")

    torch.manual_seed(cfg.seed)
    generator = torch.Generator().manual_seed(cfg.seed)
    preprocessing_cfg = OmegaConf.load(REPO_DIR / "config/preprocessing_config.yaml")
    artifacts_dir = REPO_DIR / preprocessing_cfg.output.artifacts_dir
    manifest = OmegaConf.load(artifacts_dir / preprocessing_cfg.output.manifest_filename)
    data_dir = (REPO_DIR / cfg.data.data_dir).resolve()
    pairs = pd.read_parquet(REPO_DIR / manifest.pairs_path)
    hard_negatives = pd.read_parquet(REPO_DIR / manifest.hard_negatives_path)
    cache = load_patch_feature_cache(
        patch_feature_cache_path(artifacts_dir, cfg.model.backbone, Path(manifest.pairs_path)),
        pairs,
        cfg.model.backbone,
        str(cfg.model.weights),
        cfg.model.image_size,
        OmegaConf.to_container(cfg.model.normalization, resolve=True),
    )
    train_features = {season: cache["splits"]["train"][season] for season in ("summer", "snow")}
    val_features = {season: cache["splits"]["val"][season] for season in ("summer", "snow")}
    negative_index = build_hard_negative_index(
        pairs, hard_negatives, data_dir, cfg.training.sampled_hard_negatives_per_anchor
    )

    device = get_device()
    model = SALADPool(
        input_dim=train_features["summer"]["cls"].shape[-1],
        num_clusters=salad_cfg.num_clusters,
        cluster_dim=salad_cfg.cluster_dim,
        token_dim=salad_cfg.token_dim,
        hidden_dim=salad_cfg.hidden_dim,
        dropout=salad_cfg.dropout,
        sinkhorn_iterations=salad_cfg.sinkhorn_iterations,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    scheduler = build_scheduler(optimizer, cfg.scheduler, cfg.training.epochs)
    criterion = InfoNCELoss(cfg.training.temperature)
    checkpoint_prefix = REPO_DIR / cfg.checkpoint.output_dir / cfg.checkpoint.run_name
    writer = SummaryWriter(log_dir=REPO_DIR / cfg.logging.tensorboard_dir / cfg.checkpoint.run_name)
    writer.add_text(TRAINING_CONFIG_TEXT, OmegaConf.to_yaml(cfg.training, resolve=True), global_step=0)
    print(
        f"Training {cfg.checkpoint.run_name} (SALAD, {model.output_dim}-D descriptor): "
        f"{len(train_features['summer']['cls']):,} train / {len(val_features['summer']['cls']):,} val pairs",
        flush=True,
    )

    start_epoch, best_score, best_epoch = 1, float("-inf"), 0
    if args.resume is not None:
        checkpoint = load_checkpoint(args.resume.expanduser().resolve(), device)
        restore_training_state(checkpoint, model, optimizer, scheduler)
        start_epoch = checkpoint.epoch + 1
        best_score, best_epoch = checkpoint.best_score, checkpoint.best_epoch

    try:
        for epoch in range(start_epoch, cfg.training.epochs + 1):
            train_metrics = train_one_epoch(
                model, optimizer, criterion, train_features["summer"], train_features["snow"],
                negative_index, cfg.training.batch_size, generator,
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
                (TRAIN_LEARNING_RATE, optimizer.param_groups[0]["lr"]),
                (VALIDATION_RETRIEVAL_MEAN_RECALL_AT_1, validation_metrics.mean_recall_at_1),
                (VALIDATION_RETRIEVAL_SUMMER_TO_SNOW_RECALL_AT_1, validation_metrics.summer_to_snow[RECALL_AT_1]),
                (VALIDATION_RETRIEVAL_SNOW_TO_SUMMER_RECALL_AT_1, validation_metrics.snow_to_summer[RECALL_AT_1]),
                (VALIDATION_RETRIEVAL_SUMMER_TO_SNOW_RECALL_AT_5, validation_metrics.summer_to_snow[RECALL_AT_5]),
                (VALIDATION_RETRIEVAL_SNOW_TO_SUMMER_RECALL_AT_5, validation_metrics.snow_to_summer[RECALL_AT_5]),
            ]:
                writer.add_scalar(tag, value, epoch)
            print(
                f"Epoch {epoch}/{cfg.training.epochs} loss={train_metrics.total_loss:.4f} "
                f"mean_recall@1={validation_metrics.mean_recall_at_1:.4f}",
                flush=True,
            )
            scheduler.step()
            checkpoint_args = (
                model, optimizer, scheduler, epoch, train_metrics, validation_metrics,
                best_score, best_epoch, checkpoint_config,
            )
            if is_best:
                save_checkpoint(checkpoint_prefix.with_name(f"{checkpoint_prefix.name}_best.pt"), *checkpoint_args)
            save_checkpoint(checkpoint_prefix.with_name(f"{checkpoint_prefix.name}_last.pt"), *checkpoint_args)
    finally:
        writer.close()

if __name__ == "__main__":
    main()
