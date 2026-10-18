import argparse
import os
from pathlib import Path
import sys

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from omegaconf import OmegaConf

from src.preprocessing.data_cleaner import load_pairs
from src.preprocessing.hard_negative_mining import precompute_hard_negatives
from src.preprocessing.schema import Pairs
from src.preprocessing.splits import assign_geographic_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create geographic SeasoNet splits and hard-negative artifacts."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_DIR / "config/preprocessing_config.yaml",
        help="Path to the preprocessing configuration.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("SEASONET_DIR", REPO_DIR / "data")),
        help="SeasoNet dataset directory. Overrides SEASONET_DIR.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    data_dir = args.data_dir.resolve()
    artifacts_dir = REPO_DIR / cfg.output.artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    split_cfg = cfg.geographic_split
    hard_negative_cfg = cfg.hard_negatives
    settings = (
        f"clouds{cfg.cleaning.max_cloud_cover}"
        f"_block{split_cfg.block_size_m // 1000}km"
        f"_buffer{split_cfg.buffer_m}m"
    )

    raw_pairs = load_pairs(data_dir, max_cloud_cover=cfg.cleaning.max_cloud_cover)
    pairs = assign_geographic_splits(
        raw_pairs,
        block_size_m=split_cfg.block_size_m,
        buffer_m=split_cfg.buffer_m,
        reference_latitude=split_cfg.reference_latitude,
        fractions=dict(split_cfg.fractions),
        priority=dict(split_cfg.priority),
        seed=split_cfg.seed,
    )

    pairs_path = artifacts_dir / f"pairs_{settings}.parquet"
    pairs.to_parquet(pairs_path, index=False)

    train_pairs = pairs[pairs[Pairs.SPLIT] == "train"].reset_index(drop=True)
    if train_pairs.empty:
        raise ValueError("Geographic split produced no training pairs")

    hard_negatives = precompute_hard_negatives(
        train_pairs,
        k=hard_negative_cfg.k,
        candidate_count=hard_negative_cfg.candidate_count,
        exclusion_radius_m=hard_negative_cfg.exclusion_radius_m,
    )
    hard_negatives_path = artifacts_dir / (
        f"hard_negatives_{settings}_k{hard_negative_cfg.k}"
        f"_exclusion{hard_negative_cfg.exclusion_radius_m}m.parquet"
    )
    hard_negatives.to_parquet(hard_negatives_path, index=False)

    manifest_path = artifacts_dir / cfg.output.manifest_filename
    OmegaConf.save(
        config=OmegaConf.create({
            "pairs_path": str(pairs_path.relative_to(REPO_DIR)),
            "hard_negatives_path": str(hard_negatives_path.relative_to(REPO_DIR)),
        }),
        f=manifest_path,
    )

    print(f"Pairs after cloud filter: {len(raw_pairs):,}")
    print(f"Pairs after geographic split: {len(pairs):,}")
    print(pairs[Pairs.SPLIT].value_counts().sort_index().to_string())
    print(f"Saved pairs: {pairs_path.relative_to(REPO_DIR)}")
    print(f"Saved hard negatives: {hard_negatives_path.relative_to(REPO_DIR)}")
    print(f"Saved manifest: {manifest_path.relative_to(REPO_DIR)}")


if __name__ == "__main__":
    main()
