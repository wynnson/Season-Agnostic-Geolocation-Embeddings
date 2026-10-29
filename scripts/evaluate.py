from pathlib import Path
import sys

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import pandas as pd
import torch

from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader

from src.datasets.seasonet import SeasoNetDataset
from src.models.backbone import load_frozen_backbone, load_model
from src.models.preprocessing import embed_images
from src.models.salad import SALADPool
from src.training.checkpoint import load_checkpoint
from src.training.evaluation import evaluate_cross_season_retrieval
from src.utils.device import get_device


class SALADDescriptor(nn.Module):
    def __init__(self, backbone: nn.Module, pool: SALADPool):
        super().__init__()
        self.backbone = backbone
        self.pool = pool

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(images)
        return self.pool(features["x_norm_clstoken"], features["x_norm_patchtokens"])


def load_trained_model(checkpoint_path: Path, device: torch.device):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = load_checkpoint(checkpoint_path, device)
    train_cfg = OmegaConf.create(checkpoint.config)
    if "dustbin_score" in checkpoint.model_state_dict:
        salad_cfg = train_cfg.model.salad
        pool = SALADPool(
            input_dim=checkpoint.model_state_dict["token_features.0.weight"].shape[1],
            num_clusters=salad_cfg.num_clusters,
            cluster_dim=salad_cfg.cluster_dim,
            token_dim=salad_cfg.token_dim,
            hidden_dim=salad_cfg.hidden_dim,
            dropout=salad_cfg.dropout,
            sinkhorn_iterations=salad_cfg.sinkhorn_iterations,
        )
        pool.load_state_dict(checkpoint.model_state_dict)
        backbone = load_frozen_backbone(
            train_cfg.model.backbone,
            train_cfg.model.repository,
            train_cfg.model.weights,
        )
        model = SALADDescriptor(backbone, pool).to(device)
        model.eval()
        return model, train_cfg.model

    model = load_model(
        model_name=train_cfg.model.backbone,
        repository=train_cfg.model.repository,
        tuning=train_cfg.model.tuning,
        embedding_dim=train_cfg.model.projection_head.embedding_dim,
        hidden_dim=train_cfg.model.projection_head.hidden_dim,
        projection_dropout=train_cfg.model.projection_head.dropout,
        season_classes=train_cfg.model.season_classifier.num_classes,
        projection_architecture=train_cfg.model.projection_head.get("architecture", "mlp"),
        weights=train_cfg.model.weights,
        lora_config=train_cfg.model.get("lora"),
        patch_attention_config=(
            train_cfg.model.patch_attention
            if "patch_attention" in train_cfg.model
            else None
        ),
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

    test_dataset = SeasoNetDataset(
        pairs,
        split="test",
        data_dir=data_dir,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.evaluation.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
    )

    device = get_device()

    match cfg.evaluation.mode:
        case "trained" | "lora":
            checkpoint_path = REPO_DIR / cfg.evaluation.checkpoint_path
            model, model_cfg = load_trained_model(checkpoint_path, device)
            model_type = "SALAD descriptor" if isinstance(model, SALADDescriptor) else "Trained season-agnostic model"
            model_label = f"{model_type} ({checkpoint_path.name})"
        case "frozen":
            model = load_frozen_backbone(
                cfg.model.backbone,
                cfg.model.repository,
                cfg.model.weights,
            ).to(device)
            model.eval()
            model_cfg = cfg.model
            model_label = f"Frozen {cfg.model.backbone}"
        case mode:
            raise ValueError(f"Unsupported evaluation mode: {mode}")

    def embed_fn(images: torch.Tensor) -> torch.Tensor:
        return embed_images(
            model,
            images.to(device),
            model_cfg.image_size,
            tuple(model_cfg.normalization.mean),
            tuple(model_cfg.normalization.std),
        )

    metrics = evaluate_cross_season_retrieval(
        test_loader,
        embed_fn=embed_fn,
        chunk_size=cfg.evaluation.retrieval_chunk_size,
        log_every=cfg.logging.log_every,
    )

    print(f"{model_label} evaluation on geographic test split ({len(test_dataset):,} pairs)")
    print("Summer -> Snow:", metrics.summer_to_snow)
    print("Snow -> Summer:", metrics.snow_to_summer)


if __name__ == "__main__":
    main()
