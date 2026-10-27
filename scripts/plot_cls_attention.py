import argparse
from pathlib import Path
import sys

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

from omegaconf import OmegaConf
from torch.nn import functional as F

from scripts.evaluate import load_trained_model
from src.datasets.seasonet import SeasoNetDataset
from src.models.preprocessing import preprocess_images
from src.preprocessing.schema import Pairs
from src.utils.device import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot last-block CLS self-attention for trained checkpoints.")
    parser.add_argument("loc_id", help="Location id from the pairs table.")
    parser.add_argument("output", type=Path, help="Image path to write.")
    parser.add_argument("--run", action="append", required=True, help="LABEL=CHECKPOINT_PATH; repeat per model.")
    parser.add_argument("--val-locations", type=int, default=1000, help="Validation locations for the mean r.")
    return parser.parse_args()


def cls_attention(model, image: torch.Tensor, model_cfg) -> torch.Tensor:
    attention = model.backbone.blocks[-1].attn
    captured = {}

    def capture(module, args, kwargs):
        captured["tokens"], captured["rope"] = args[0], kwargs.get("rope")

    handle = attention.register_forward_pre_hook(capture, with_kwargs=True)
    images = preprocess_images(
        image[None],
        model_cfg.image_size,
        tuple(model_cfg.normalization.mean),
        tuple(model_cfg.normalization.std),
    )
    try:
        with torch.inference_mode():
            model.backbone(images)
    finally:
        handle.remove()

    with torch.inference_mode():
        tokens = captured["tokens"]
        batch, count, dim = tokens.shape
        qkv = attention.qkv(tokens).reshape(batch, count, 3, attention.num_heads, dim // attention.num_heads)
        q, k, v = (t.transpose(1, 2) for t in qkv.unbind(2))
        if captured["rope"] is not None:
            q, k = attention.apply_rope(q, k, captured["rope"])
        weights = torch.softmax(q @ k.transpose(-2, -1) * attention.scale, dim=-1)
        # The recomputed weights must reproduce the block's own attention output.
        if not torch.allclose(weights @ v, F.scaled_dot_product_attention(q, k, v), atol=1e-4):
            raise RuntimeError("Recomputed attention does not match the backbone")

    first_patch = 1 + getattr(model.backbone, "n_storage_tokens", 0)
    cls_to_patches = weights[0, :, 0, first_patch:].mean(dim=0).cpu()
    side = int(cls_to_patches.numel() ** 0.5)
    # Renormalized over patches, so maps are comparable across models.
    return (cls_to_patches / cls_to_patches.sum()).reshape(side, side)


def correlation(first: torch.Tensor, second: torch.Tensor) -> float:
    return torch.corrcoef(torch.stack((first.flatten(), second.flatten())))[0, 1].item()


def display_rgb(image: torch.Tensor):
    # Per-image 2-98 percentile stretch, for display only.
    rgb = image.permute(1, 2, 0)
    low, high = torch.quantile(rgb.flatten(), torch.tensor([0.02, 0.98]))
    return ((rgb - low) / (high - low)).clamp(0, 1).numpy()


def main() -> None:
    args = parse_args()
    preprocessing_cfg = OmegaConf.load(REPO_DIR / "config/preprocessing_config.yaml")
    artifacts_dir = REPO_DIR / preprocessing_cfg.output.artifacts_dir
    manifest = OmegaConf.load(artifacts_dir / preprocessing_cfg.output.manifest_filename)
    data_dir = (REPO_DIR / OmegaConf.load(REPO_DIR / "config/train_config.yaml").data.data_dir).resolve()
    pairs = pd.read_parquet(REPO_DIR / manifest.pairs_path)

    location = pairs.loc[pairs[Pairs.LOC_ID].astype(str) == args.loc_id]
    if location.empty:
        raise ValueError(f"Location {args.loc_id} is not in the pairs table")
    split = location[Pairs.SPLIT].item()
    dataset = SeasoNetDataset(pairs, split=split, data_dir=data_dir)
    sample = dataset[dataset.pairs.index[dataset.pairs[Pairs.LOC_ID] == args.loc_id][0]]
    images = {"Summer": sample.anchor, "Snow": sample.positive}

    val_dataset = SeasoNetDataset(pairs, split="val", data_dir=data_dir)
    rows = torch.randperm(len(val_dataset), generator=torch.Generator().manual_seed(0))[:args.val_locations].tolist()

    device = get_device()
    maps, mean_correlations = {}, {}
    for run in args.run:
        label, _, checkpoint = run.partition("=")
        model, model_cfg = load_trained_model(REPO_DIR / checkpoint, device)
        maps[label] = {season: cls_attention(model, image.to(device), model_cfg) for season, image in images.items()}
        mean_correlations[label] = sum(
            correlation(
                cls_attention(model, val_dataset[row].anchor.to(device), model_cfg),
                cls_attention(model, val_dataset[row].positive.to(device), model_cfg),
            )
            for row in rows
        ) / len(rows)
        print(
            f"{label}: summer/snow attention r = {correlation(maps[label]['Summer'], maps[label]['Snow']):.2f} here, "
            f"{mean_correlations[label]:.2f} mean over {len(rows)} val locations",
            flush=True,
        )

    vmax = max(attention_map.max().item() for per_run in maps.values() for attention_map in per_run.values())
    figure, axes = plt.subplots(1 + len(maps), 2, figsize=(10.5, 5 * (1 + len(maps))), constrained_layout=True)
    for column, (season, image) in enumerate(images.items()):
        rgb = display_rgb(image)
        axes[0, column].imshow(rgb)
        axes[0, column].set_title(f"{season} RGB", fontsize=12)
        for row, (label, per_run) in enumerate(maps.items(), start=1):
            heat = F.interpolate(per_run[season][None, None], size=rgb.shape[:2], mode="bilinear", align_corners=False)
            axes[row, column].imshow(rgb)
            heatmap = axes[row, column].imshow(heat[0, 0].numpy(), cmap="magma", vmin=0, vmax=vmax, alpha=0.65)
            axes[row, column].set_title(f"{label}: {season}", fontsize=12)
    for row, label in enumerate(maps, start=1):
        axes[row, 0].text(
            -0.04,
            0.5,
            f"{label}\nsummer/snow map r = {correlation(maps[label]['Summer'], maps[label]['Snow']):.2f}"
            f"\n(mean over {len(rows)} val: {mean_correlations[label]:.2f})",
            transform=axes[row, 0].transAxes,
            rotation=90,
            ha="right",
            va="center",
            fontsize=11,
        )
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    figure.colorbar(heatmap, ax=axes, shrink=0.5, label="CLS-to-patch attention weight (last block, head mean, sums to 1)")
    figure.suptitle(f"Location {args.loc_id} ({split}): last-block CLS self-attention", fontsize=13)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=120)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
