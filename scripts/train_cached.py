"""Train frozen-backbone heads from cached CLS features, with optional patch attention."""
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
from src.models.backbone import load_model
from src.preprocessing.schema import Pairs
from src.training.checkpoint import load_checkpoint, restore_training_state, save_checkpoint
from src.training.evaluation import retrieval_recall_at_k
from src.training.feature_cache import (
    SEASONS,
    embed_features,
    feature_cache_path,
    forward_heads,
    load_feature_cache,
    load_patch_feature_cache,
    patch_feature_cache_path,
)
from src.training.losses.attention import symmetric_js_divergence
from src.training.losses.info_nce import InfoNCELoss
from src.training.scheduler import build_scheduler
from src.training.schema import (
    HPARAM_BEST_MEAN_RECALL_AT_1,
    RECALL_AT_1,
    RECALL_AT_5,
    TRAIN_CONTRASTIVE_HEAD_LOSS,
    TRAIN_LEARNING_RATE,
    TRAIN_SEASON_CLASSIFIER_ACCURACY,
    TRAIN_SEASON_CLASSIFIER_LOSS,
    TRAIN_SNOW_CLASSIFIER_ACCURACY,
    TRAIN_SUMMER_CLASSIFIER_ACCURACY,
    TRAIN_TOTAL_LOSS,
    TRAINING_CONFIG_TEXT,
    VALIDATION_RETRIEVAL_MEAN_RECALL_AT_1,
    VALIDATION_RETRIEVAL_SNOW_TO_SUMMER_RECALL_AT_1,
    VALIDATION_RETRIEVAL_SNOW_TO_SUMMER_RECALL_AT_5,
    VALIDATION_RETRIEVAL_SUMMER_TO_SNOW_RECALL_AT_1,
    VALIDATION_RETRIEVAL_SUMMER_TO_SNOW_RECALL_AT_5,
    TrainingMetrics,
    ValidationMetrics,
)
from src.utils.device import get_device


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Train frozen-backbone heads on cached DINO features.")
    parser.add_argument("--resume", type=Path, help="Checkpoint path to resume from.")
    return parser.parse_known_args()


def build_hard_negative_index(
    train_pairs: pd.DataFrame,
    hard_negatives: pd.DataFrame,
    data_dir: Path,
    k: int,
) -> torch.Tensor | None:
    if k == 0:
        return None

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


def make_feature_sets(cache: dict, split: str, patch_attention_enabled: bool) -> dict[str, dict[str, torch.Tensor]]:
    if patch_attention_enabled:
        return {season: cache["splits"][split][season] for season in SEASONS}
    return {
        season: {"cls": cache["splits"][split][season]}
        for season in SEASONS
    }


def batch_features(
    features: dict[str, torch.Tensor],
    indices: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    cls_features = features["cls"][indices].to(device=device, dtype=torch.float32)
    patch_features = features.get("patches")
    if patch_features is None:
        return cls_features, None
    return cls_features, patch_features[indices].to(device=device, dtype=torch.float32)


@torch.inference_mode()
def validate(
    model,
    summer: dict[str, torch.Tensor],
    snow: dict[str, torch.Tensor],
    chunk_size: int,
) -> ValidationMetrics:
    model.eval()
    device = next(model.parameters()).device

    def embed_in_batches(features: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([
            embed_features(
                model,
                *batch_features(
                    features,
                    torch.arange(start, min(start + chunk_size, len(features["cls"]))),
                    device,
                ),
            )
            for start in range(0, len(features["cls"]), chunk_size)
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
    summer: dict[str, torch.Tensor],
    snow: dict[str, torch.Tensor],
    negative_index: torch.Tensor | None,
    batch_size: int,
    season_loss_weight: float,
    attention_consistency_weight: float,
    grl_scale: float,
    generator: torch.Generator,
) -> TrainingMetrics:
    model.train()
    device = next(model.parameters()).device
    order = torch.randperm(len(summer["cls"]), generator=generator)
    num_batches = len(order) // batch_size
    totals = torch.zeros(7, device=device)
    progress = tqdm(range(num_batches), desc="Cached training", unit="batch", leave=False)

    for batch_number in progress:
        indices = order[batch_number * batch_size:(batch_number + 1) * batch_size]
        summer_cls, summer_patches = batch_features(summer, indices, device)
        snow_cls, snow_patches = batch_features(snow, indices, device)
        use_attention = model.patch_attention is not None
        summer_output = forward_heads(
            model,
            summer_cls,
            grl_scale,
            summer_patches,
            return_attention=use_attention,
        )
        snow_output = forward_heads(
            model,
            snow_cls,
            grl_scale,
            snow_patches,
            return_attention=use_attention,
        )
        if use_attention:
            summer_embeddings, summer_logits, summer_attention = summer_output
            snow_embeddings, snow_logits, snow_attention = snow_output
            attention_consistency_loss = symmetric_js_divergence(summer_attention, snow_attention)
        else:
            summer_embeddings, summer_logits = summer_output
            snow_embeddings, snow_logits = snow_output
            attention_consistency_loss = torch.zeros((), device=device)

        if negative_index is None:
            summer_negative_embeddings = None
            snow_negative_embeddings = None
        else:
            negatives = negative_index[indices]
            summer_negative_embeddings = embed_features(model, *batch_features(summer, negatives, device))
            snow_negative_embeddings = embed_features(model, *batch_features(snow, negatives, device))

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
        total_loss = (
            contrastive_loss
            + season_loss_weight * season_loss
            + attention_consistency_weight * attention_consistency_loss
        )

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        totals += torch.stack((
            total_loss,
            contrastive_loss,
            season_loss,
            attention_consistency_loss,
            season_classifier_accuracy,
            summer_classifier_accuracy,
            snow_classifier_accuracy,
        )).detach()
        progress.set_postfix(
            contrastive=f"{contrastive_loss.item():.4f}",
            attention=f"{attention_consistency_loss.item():.4f}",
        )

    total, contrastive, season, attention, classifier, summer_accuracy, snow_accuracy = (totals / num_batches).tolist()
    return TrainingMetrics(
        total_loss=total,
        contrastive_loss=contrastive,
        season_loss=season,
        attention_consistency_loss=attention,
        season_classifier_accuracy=classifier,
        summer_classifier_accuracy=summer_accuracy,
        snow_classifier_accuracy=snow_accuracy,
    )


def attention_entropy(attention: torch.Tensor) -> float:
    return (-(attention * attention.clamp_min(torch.finfo(attention.dtype).eps).log()).sum(dim=-1)).mean().item()


def main() -> None:
    args, overrides = parse_args()
    cfg = OmegaConf.merge(
        OmegaConf.load(REPO_DIR / "config/train_config.yaml"),
        OmegaConf.from_dotlist(overrides),
    )
    checkpoint_config = OmegaConf.to_container(cfg, resolve=True)
    if cfg.model.tuning != "frozen":
        raise ValueError("Cached feature training is only valid with model.tuning: frozen")

    patch_attention_enabled = cfg.model.patch_attention.enabled
    torch.manual_seed(cfg.seed)
    generator = torch.Generator().manual_seed(cfg.seed)
    preprocessing_cfg = OmegaConf.load(REPO_DIR / "config/preprocessing_config.yaml")
    artifacts_dir = REPO_DIR / preprocessing_cfg.output.artifacts_dir
    manifest = OmegaConf.load(artifacts_dir / preprocessing_cfg.output.manifest_filename)
    data_dir = (REPO_DIR / cfg.data.data_dir).resolve()
    pairs = pd.read_parquet(REPO_DIR / manifest.pairs_path)
    hard_negatives = pd.read_parquet(REPO_DIR / manifest.hard_negatives_path)

    if patch_attention_enabled:
        cache = load_patch_feature_cache(
            patch_feature_cache_path(artifacts_dir, cfg.model.backbone, Path(manifest.pairs_path)),
            pairs,
            cfg.model.backbone,
            str(cfg.model.weights),
            cfg.model.image_size,
            OmegaConf.to_container(cfg.model.normalization, resolve=True),
        )
    else:
        cache = load_feature_cache(
            feature_cache_path(artifacts_dir, cfg.model.backbone, Path(manifest.pairs_path)),
            pairs,
        )

    device = get_device()
    train_features = make_feature_sets(cache, "train", patch_attention_enabled)
    val_features = make_feature_sets(cache, "val", patch_attention_enabled)
    negative_index = build_hard_negative_index(
        pairs,
        hard_negatives,
        data_dir,
        cfg.training.sampled_hard_negatives_per_anchor,
    )
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
        patch_attention_config=cfg.model.patch_attention if patch_attention_enabled else None,
    ).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = build_scheduler(optimizer, cfg.scheduler, cfg.training.epochs)
    criterion = InfoNCELoss(cfg.training.temperature)
    checkpoint_prefix = REPO_DIR / cfg.checkpoint.output_dir / cfg.checkpoint.run_name
    writer = SummaryWriter(log_dir=REPO_DIR / cfg.logging.tensorboard_dir / cfg.checkpoint.run_name)
    writer.add_text(TRAINING_CONFIG_TEXT, OmegaConf.to_yaml(cfg.training, resolve=True), global_step=0)

    start_epoch, best_score, best_epoch = 1, float("-inf"), 0
    if args.resume is not None:
        checkpoint = load_checkpoint(args.resume.expanduser().resolve(), device)
        restore_training_state(checkpoint, model, optimizer, scheduler)
        start_epoch = checkpoint.epoch + 1
        best_score = checkpoint.best_score
        best_epoch = checkpoint.best_epoch

    try:
        for epoch in range(start_epoch, cfg.training.epochs + 1):
            train_metrics = train_one_epoch(
                model,
                optimizer,
                criterion,
                train_features["summer"],
                train_features["snow"],
                negative_index,
                cfg.training.batch_size,
                cfg.training.season_loss_weight,
                cfg.training.attention_consistency_weight,
                cfg.training.grl_scale,
                generator,
            )
            validation_metrics = validate(
                model,
                val_features["summer"],
                val_features["snow"],
                cfg.training.retrieval_chunk_size,
            )
            is_best = validation_metrics.mean_recall_at_1 > best_score
            if is_best:
                best_score, best_epoch = validation_metrics.mean_recall_at_1, epoch

            for tag, value in [
                (TRAIN_TOTAL_LOSS, train_metrics.total_loss),
                (TRAIN_CONTRASTIVE_HEAD_LOSS, train_metrics.contrastive_loss),
                (TRAIN_SEASON_CLASSIFIER_LOSS, train_metrics.season_loss),
                ("train/patch_attention/consistency_loss", train_metrics.attention_consistency_loss),
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
                with torch.inference_mode():
                    for season in SEASONS:
                        sample_indices = torch.arange(min(cfg.training.batch_size, len(val_features[season]["cls"])))
                        sample_cls, sample_patches = batch_features(val_features[season], sample_indices, device)
                        _, attention = model.embed_features(sample_cls, sample_patches, return_attention=True)
                        writer.add_scalar(f"patch_attention/{season}_entropy", attention_entropy(attention), epoch)
                        writer.add_scalar(
                            f"patch_attention/{season}_max_weight",
                            attention.max(dim=-1).values.mean().item(),
                            epoch,
                        )
                    writer.add_scalar("patch_attention/residual_scale", model.patch_attention.residual_scale.item(), epoch)

            print(
                f"Epoch {epoch}/{cfg.training.epochs} "
                f"loss={train_metrics.total_loss:.4f} "
                f"contrastive={train_metrics.contrastive_loss:.4f} "
                f"attention={train_metrics.attention_consistency_loss:.4f} "
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
    finally:
        writer.close()


if __name__ == "__main__":
    main()
