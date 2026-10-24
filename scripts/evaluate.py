from pathlib import Path
import sys

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import pandas as pd
import torch

from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.datasets.seasonet import SeasoNetDataset
from src.models.dinov2 import backbone_weights_path, embed_dinov2_images, load_dinov2, load_frozen_dinov2
from src.training.checkpoint import load_checkpoint
from src.training.evaluation import evaluate_cross_season_retrieval
from src.utils.device import get_device


def load_trained_model(checkpoint_path: Path, device: torch.device):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = load_checkpoint(checkpoint_path, device)
    train_cfg = OmegaConf.create(checkpoint.config)
    model = load_dinov2(
        model_name=train_cfg.model.backbone,
        repository=train_cfg.model.repository,
        tuning=train_cfg.model.tuning,
        embedding_dim=train_cfg.model.projection_head.embedding_dim,
        hidden_dim=train_cfg.model.projection_head.hidden_dim,
        projection_dropout=train_cfg.model.projection_head.dropout,
        season_classes=train_cfg.model.season_classifier.num_classes,
        projection_architecture=train_cfg.model.projection_head.get("architecture", "mlp"),
        weights=backbone_weights_path(train_cfg.model, REPO_DIR),
    ).to(device)
    model.load_state_dict(checkpoint.model_state_dict)
    model.eval()
    return model, train_cfg.model


def main():
    cfg = OmegaConf.merge(
        OmegaConf.load(REPO_DIR / "config/evaluation_config.yaml"),
        OmegaConf.from_dotlist(sys.argv[1:]),
    )
    preprocessing_cfg = OmegaConf.load(REPO_DIR / "config/preprocessing_config.yaml")
    manifest_path = (
        REPO_DIR
        / preprocessing_cfg.output.artifacts_dir
        / preprocessing_cfg.output.manifest_filename
    )
    artifact_manifest = OmegaConf.load(manifest_path)
    torch.manual_seed(cfg.seed)

    data_dir = (REPO_DIR / cfg.data.data_dir).resolve()
    pairs = pd.read_parquet(REPO_DIR / artifact_manifest.pairs_path)

    val_dataset = SeasoNetDataset(pairs, split="val", data_dir=data_dir)
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.evaluation.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
    )

    device = get_device()

    match cfg.evaluation.mode:
        case "trained":
            checkpoint_path = REPO_DIR / cfg.evaluation.checkpoint_path
            model, model_cfg = load_trained_model(checkpoint_path, device)
            model_label = f"Trained DINOv2 ({checkpoint_path.name})"
        case "frozen":
            model = load_frozen_dinov2(
                cfg.model.backbone,
                cfg.model.repository,
                backbone_weights_path(cfg.model, REPO_DIR),
            ).to(device)
            model.eval()
            model_cfg = cfg.model
            model_label = f"Frozen {cfg.model.backbone}"
        case "lora":
            raise NotImplementedError("LoRA evaluation is not implemented yet")
        case mode:
            raise ValueError(f"Unsupported evaluation mode: {mode}")

    def embed_fn(images: torch.Tensor) -> torch.Tensor:
        return embed_dinov2_images(
            model,
            images.to(device),
            model_cfg.image_size,
            tuple(model_cfg.normalization.mean),
            tuple(model_cfg.normalization.std),
        )

    metrics = evaluate_cross_season_retrieval(
        val_loader,
        embed_fn=embed_fn,
        chunk_size=cfg.evaluation.retrieval_chunk_size,
        log_every=cfg.logging.log_every,
    )

    print(f"{model_label} evaluation on geographic validation split ({len(val_dataset):,} pairs)")
    print("Summer -> Snow:", metrics.summer_to_snow)
    print("Snow -> Summer:", metrics.snow_to_summer)


if __name__ == "__main__":
    main()
