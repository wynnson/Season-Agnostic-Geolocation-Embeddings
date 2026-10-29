"""Train LoRA blocks and heads on cached DINOv3 tokens from scripts/extract_tokens.py.

Only the blocks from the first LoRA block on run, so model.tuning: frozen and lora share the cache
and differ only in the adapters. Same objective and checkpoint format as scripts/train.py.
Config values can be overridden from the command line, for example:
    uv run python scripts/train_cached_lora.py model.tuning=frozen checkpoint.run_name=frozen_matched
"""
import argparse
from pathlib import Path
import sys

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import numpy as np
import pandas as pd
import torch

from omegaconf import OmegaConf
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from scripts.train_cached import build_hard_negative_index
from src.models.backbone import features_from_block, first_lora_block, load_model
from src.models.gradient_reversal import gradient_reverse
from src.training.checkpoint import load_checkpoint, restore_training_state, save_checkpoint
from src.training.evaluation import retrieval_recall_at_k
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
from src.training.token_cache import gather_tokens, load_token_cache, token_cache_dir
from src.utils.device import get_device


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Train LoRA blocks and heads on cached backbone tokens.")
    parser.add_argument("--resume", type=Path, help="Checkpoint path to resume from.")
    return parser.parse_known_args()


def embed_tokens(model, tokens: torch.Tensor, block: int, return_attention: bool = False):
    # Mirrors SeasonAgnosticModel.forward_embeddings from `block` on.
    cls_features, patch_features = features_from_block(model.backbone, tokens, block)
    if model.patch_attention is None:
        cls_features, patch_features = model.backbone.head(cls_features), None
    return model.embed_features(cls_features, patch_features, return_attention=return_attention)


@torch.inference_mode()
def validate(model, summer: np.ndarray, snow: np.ndarray, block: int, chunk_size: int) -> ValidationMetrics:
    model.eval()
    device = next(model.parameters()).device

    def embed_in_batches(tokens: np.ndarray) -> torch.Tensor:
        return torch.cat([
            embed_tokens(model, gather_tokens(tokens, torch.arange(start, min(start + chunk_size, len(tokens)))).to(device), block)
            for start in range(0, len(tokens), chunk_size)
        ])

    summer_embeddings = embed_in_batches(summer)
    snow_embeddings = embed_in_batches(snow)
    return ValidationMetrics(
        summer_to_snow=retrieval_recall_at_k(summer_embeddings, snow_embeddings, chunk_size=chunk_size),
        snow_to_summer=retrieval_recall_at_k(snow_embeddings, summer_embeddings, chunk_size=chunk_size),
    )


def train_one_epoch(
    model,
    optimizer: torch.optim.Optimizer,
    criterion: InfoNCELoss,
    summer: np.ndarray,
    snow: np.ndarray,
    negative_index: torch.Tensor | None,
    block: int,
    batch_size: int,
    season_loss_weight: float,
    grl_scale: float,
    generator: torch.Generator,
) -> TrainingMetrics:
    model.train()
    # Eval mode keeps DINOv3's RoPE deterministic, which the cached tokens assume.
    model.backbone.eval()
    device = next(model.parameters()).device
    order = torch.randperm(len(summer), generator=generator)
    num_batches = len(order) // batch_size  # drop_last, as in scripts/train.py

    totals = torch.zeros(6, device=device)
    progress = tqdm(range(num_batches), desc="Cached LoRA training", unit="batch", leave=False)
    for b in progress:
        idx = order[b * batch_size:(b + 1) * batch_size]
        rows = idx if negative_index is None else torch.cat((idx, negative_index[idx].flatten()))
        # Anchors and their hard negatives share one pass through the trainable blocks.
        summer_all = embed_tokens(model, gather_tokens(summer, rows).to(device), block)
        snow_all = embed_tokens(model, gather_tokens(snow, rows).to(device), block)
        summer_embeddings, snow_embeddings = summer_all[:batch_size], snow_all[:batch_size]
        if negative_index is None:
            summer_negative_embeddings = None
            snow_negative_embeddings = None
        else:
            summer_negative_embeddings = summer_all[batch_size:].reshape(batch_size, negative_index.shape[1], -1)
            snow_negative_embeddings = snow_all[batch_size:].reshape(batch_size, negative_index.shape[1], -1)

        contrastive_loss = (
            criterion(summer_embeddings, snow_embeddings, snow_negative_embeddings)
            + criterion(snow_embeddings, summer_embeddings, summer_negative_embeddings)
        ) / 2

        season_labels = torch.cat((
            torch.zeros(batch_size, dtype=torch.long, device=device),
            torch.ones(batch_size, dtype=torch.long, device=device),
        ))
        season_logits = model.season_classifier(
            gradient_reverse(torch.cat((summer_embeddings, snow_embeddings)), grl_scale)
        )
        season_loss = F.cross_entropy(season_logits, season_labels)
        season_predictions = season_logits.argmax(dim=1)
        season_classifier_accuracy = (season_predictions == season_labels).float().mean()
        summer_classifier_accuracy = (season_predictions[:batch_size] == 0).float().mean()
        snow_classifier_accuracy = (season_predictions[batch_size:] == 1).float().mean()
        total_loss = (
            contrastive_loss
            + season_loss_weight * season_loss
        )

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

    total, contrastive, season, classifier, summer_accuracy, snow_accuracy = (
        totals / num_batches
    ).tolist()
    return TrainingMetrics(
        total_loss=total,
        contrastive_loss=contrastive,
        season_loss=season,
        season_classifier_accuracy=classifier,
        summer_classifier_accuracy=summer_accuracy,
        snow_classifier_accuracy=snow_accuracy,
    )


def main():
    args, overrides = parse_args()
    cfg = OmegaConf.merge(
        OmegaConf.load(REPO_DIR / "config/train_config.yaml"),
        OmegaConf.from_dotlist(overrides),
    )
    checkpoint_config = OmegaConf.to_container(cfg, resolve=True)
    if cfg.model.tuning not in ("frozen", "lora"):
        raise ValueError("Cached tokens are only valid with model.tuning: frozen or lora")
    patch_attention_enabled = cfg.model.patch_attention.enabled

    torch.manual_seed(cfg.seed)
    generator = torch.Generator().manual_seed(cfg.seed)

    preprocessing_cfg = OmegaConf.load(REPO_DIR / "config/preprocessing_config.yaml")
    artifacts_dir = REPO_DIR / preprocessing_cfg.output.artifacts_dir
    manifest = OmegaConf.load(artifacts_dir / preprocessing_cfg.output.manifest_filename)
    data_dir = (REPO_DIR / cfg.data.data_dir).resolve()
    pairs = pd.read_parquet(REPO_DIR / manifest.pairs_path)
    hard_negatives = pd.read_parquet(REPO_DIR / manifest.hard_negatives_path)

    block = first_lora_block(list(cfg.model.lora.target_modules))
    tokens = load_token_cache(
        token_cache_dir(artifacts_dir, cfg.model.backbone, block, Path(manifest.pairs_path)),
        pairs,
        cfg.model.backbone,
        block,
    )
    negative_index = build_hard_negative_index(
        pairs, hard_negatives, data_dir, cfg.training.sampled_hard_negatives_per_anchor
    )

    device = get_device()
    # Full model so checkpoints load in evaluate.py and train.py.
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
        lora_config=cfg.model.lora,
        patch_attention_config=cfg.model.patch_attention if patch_attention_enabled else None,
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

    print(f"Training {cfg.checkpoint.run_name} ({cfg.model.tuning}) on tokens entering block {block}: "
          f"{len(tokens['train']['summer']):,} train / {len(tokens['val']['summer']):,} val pairs", flush=True)

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
                tokens["train"]["summer"], tokens["train"]["snow"], negative_index, block,
                batch_size=cfg.training.batch_size,
                season_loss_weight=cfg.training.season_loss_weight,
                grl_scale=cfg.training.grl_scale,
                generator=generator,
            )
            validation_metrics = validate(
                model, tokens["val"]["summer"], tokens["val"]["snow"], block, cfg.training.retrieval_chunk_size
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
            if patch_attention_enabled:
                writer.add_scalar("patch_attention/residual_scale", model.patch_attention.residual_scale.item(), epoch)

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
