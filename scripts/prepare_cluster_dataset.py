import argparse
import errno
import json
from pathlib import Path
import shutil
import sys

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import pandas as pd

from omegaconf import OmegaConf

from src.preprocessing.schema import HardNegatives, Pairs
from src.training.feature_cache import SPLITS


DEFAULT_SPLITS = ("train", "val")
RGB_PATTERN = "*_10m_RGB.tif"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a SeasoNet dataset bundle for rsync.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_DIR / "artifacts/cluster_bundle",
        help="Directory containing data/ and artifacts/ for rsync.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SPLITS,
        default=DEFAULT_SPLITS,
        help="Pair splits to include. Defaults to train val.",
    )
    parser.add_argument(
        "--link-mode",
        choices=("auto", "hardlink", "copy"),
        default="auto",
        help="Use hard links when possible to avoid duplicating local data.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output directory.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print transfer size only.")
    return parser.parse_args()


def rgb_files_for_paths(data_dir: Path, relative_paths: list[str]) -> list[tuple[Path, Path]]:
    files = []
    for relative_path in relative_paths:
        source_directory = data_dir / relative_path
        matches = sorted(source_directory.glob(RGB_PATTERN))
        if not matches:
            raise FileNotFoundError(f"No {RGB_PATTERN} file in {source_directory}")
        files.extend((source, source.relative_to(data_dir)) for source in matches)
    return files


def stage_file(source: Path, destination: Path, link_mode: str) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if link_mode == "copy":
        shutil.copy2(source, destination)
        return "copy"

    try:
        destination.hardlink_to(source)
        return "hardlink"
    except OSError as error:
        if link_mode == "hardlink":
            raise RuntimeError(f"Could not hard-link {source} to {destination}: {error}") from error
        if error.errno not in {errno.EXDEV, errno.EPERM, errno.EOPNOTSUPP}:
            raise
        shutil.copy2(source, destination)
        return "copy"


def prepare_output_directory(output_dir: Path, data_dir: Path, overwrite: bool) -> None:
    output_dir = output_dir.resolve()
    if output_dir == data_dir or output_dir.is_relative_to(data_dir):
        raise ValueError("Output directory must not be inside the source data directory")
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir}. Use --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)


def main() -> None:
    args = parse_args()
    selected_splits = tuple(dict.fromkeys(args.splits))
    cfg = OmegaConf.load(REPO_DIR / "config/train_config.yaml")
    preprocessing_cfg = OmegaConf.load(REPO_DIR / "config/preprocessing_config.yaml")
    artifacts_dir = REPO_DIR / preprocessing_cfg.output.artifacts_dir
    source_manifest = OmegaConf.load(artifacts_dir / preprocessing_cfg.output.manifest_filename)
    data_dir = (REPO_DIR / cfg.data.data_dir).resolve()

    pairs = pd.read_parquet(REPO_DIR / source_manifest.pairs_path)
    selected_pairs = pairs.loc[pairs[Pairs.SPLIT].isin(selected_splits)].copy()
    if selected_pairs.empty:
        raise ValueError(f"No pairs found for splits: {selected_splits}")

    relative_paths = pd.concat((selected_pairs[Pairs.PATH_SUMMER], selected_pairs[Pairs.PATH_SNOW]))
    relative_paths = relative_paths.drop_duplicates().tolist()
    rgb_files = rgb_files_for_paths(data_dir, relative_paths)
    transfer_bytes = sum(source.stat().st_size for source, _ in rgb_files)
    print(
        f"Preparing {len(selected_pairs):,} pairs across {', '.join(selected_splits)}: "
        f"{len(relative_paths):,} image directories / {len(rgb_files):,} RGB TIFFs / "
        f"{transfer_bytes / 1e9:.1f} GB",
        flush=True,
    )
    if args.dry_run:
        return

    output_dir = args.output_dir.resolve()
    prepare_output_directory(output_dir, data_dir, args.overwrite)
    staged_by_mode = {"hardlink": 0, "copy": 0}
    for source, relative_path in rgb_files:
        mode = stage_file(source, output_dir / "data" / relative_path, args.link_mode)
        staged_by_mode[mode] += 1

    output_artifacts = output_dir / "artifacts"
    output_artifacts.mkdir()
    pairs_name = Path(str(source_manifest.pairs_path)).name
    selected_pairs.to_parquet(output_artifacts / pairs_name, index=False)

    train_ids = set(selected_pairs.loc[selected_pairs[Pairs.SPLIT] == "train", Pairs.LOC_ID].astype(str))
    hard_negatives = pd.read_parquet(REPO_DIR / source_manifest.hard_negatives_path)
    selected_hard_negatives = hard_negatives.loc[
        hard_negatives[HardNegatives.ANCHOR_LOC_ID].astype(str).isin(train_ids)
        & hard_negatives[HardNegatives.NEGATIVE_LOC_ID].astype(str).isin(train_ids)
    ]
    hard_negatives_name = "hard_negatives_train_k50.parquet"
    selected_hard_negatives.to_parquet(output_artifacts / hard_negatives_name, index=False)

    manifest = {
        "pairs_path": f"artifacts/{pairs_name}",
        "hard_negatives_path": f"artifacts/{hard_negatives_name}",
    }
    OmegaConf.save(OmegaConf.create(manifest), output_artifacts / "preprocessing_manifest.yaml")
    (output_dir / "transfer_manifest.json").write_text(json.dumps({
        "splits": selected_splits,
        "pair_count": len(selected_pairs),
        "image_directory_count": len(relative_paths),
        "rgb_tiff_count": len(rgb_files),
        "rgb_tiff_bytes": transfer_bytes,
        "hard_negative_count": len(selected_hard_negatives),
        "link_mode_requested": args.link_mode,
        "files_hardlinked": staged_by_mode["hardlink"],
        "files_copied": staged_by_mode["copy"],
    }, indent=2) + "\n")
    print(f"Cluster bundle ready: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
