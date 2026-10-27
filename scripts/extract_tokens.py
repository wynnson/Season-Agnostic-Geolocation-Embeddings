"""Cache frozen DINOv3 tokens entering the first LoRA block, for scripts/train_cached_lora.py.

The block comes from model.lora.target_modules. Config values can be overridden from the command line.
"""
from pathlib import Path
import sys

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import pandas as pd

from omegaconf import OmegaConf

from src.datasets.seasonet import SeasoNetDataset
from src.models.backbone import first_lora_block, load_frozen_backbone
from src.training.token_cache import TOKEN_SPLITS, extract_split_tokens, token_cache_dir, write_token_cache_metadata
from src.utils.device import get_device


EXTRACTION_BATCH_SIZE = 128


def main() -> None:
    cfg = OmegaConf.merge(
        OmegaConf.load(REPO_DIR / "config/train_config.yaml"),
        OmegaConf.from_dotlist(sys.argv[1:]),
    )
    preprocessing_cfg = OmegaConf.load(REPO_DIR / "config/preprocessing_config.yaml")
    artifacts_dir = REPO_DIR / preprocessing_cfg.output.artifacts_dir
    manifest = OmegaConf.load(artifacts_dir / preprocessing_cfg.output.manifest_filename)

    data_dir = (REPO_DIR / cfg.data.data_dir).resolve()
    pairs = pd.read_parquet(REPO_DIR / manifest.pairs_path)
    block = first_lora_block(list(cfg.model.lora.target_modules))
    output_dir = token_cache_dir(artifacts_dir, cfg.model.backbone, block, Path(manifest.pairs_path))
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    backbone = load_frozen_backbone(cfg.model.backbone, cfg.model.repository, cfg.model.weights).to(device)

    loc_ids = {}
    for split in TOKEN_SPLITS:
        dataset = SeasoNetDataset(pairs, split=split, data_dir=data_dir)
        print(f"Extracting {split}: {len(dataset):,} pairs, tokens entering block {block}", flush=True)
        loc_ids[split] = extract_split_tokens(
            backbone,
            dataset,
            block,
            output_dir,
            split,
            device=device,
            image_size=cfg.model.image_size,
            mean=tuple(cfg.model.normalization.mean),
            std=tuple(cfg.model.normalization.std),
            batch_size=EXTRACTION_BATCH_SIZE,
            num_workers=cfg.data.num_workers,
        )

    # Written last, so an interrupted extraction never looks complete.
    write_token_cache_metadata(output_dir, {
        "backbone": cfg.model.backbone,
        "block": block,
        "image_size": cfg.model.image_size,
        "pairs_path": manifest.pairs_path,
        "loc_ids": loc_ids,
    })
    size_gb = sum(path.stat().st_size for path in output_dir.glob("*.npy")) / 1e9
    print(f"Saved tokens: {output_dir.relative_to(REPO_DIR)} ({size_gb:.1f} GB)")


if __name__ == "__main__":
    main()
