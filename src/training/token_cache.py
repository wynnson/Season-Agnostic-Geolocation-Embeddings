"""Cache the backbone tokens entering a block, so runs that tune only later blocks skip the earlier ones.

Valid only while the earlier blocks are frozen and images are not augmented.
Arrays are disk-backed .npy files because the train split alone is ~15 GB.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from numpy.lib.format import open_memmap
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.datasets.seasonet import SeasoNetDataset
from src.models.backbone import tokens_before_block
from src.models.preprocessing import preprocess_images
from src.preprocessing.schema import Pairs
from src.training.feature_cache import SEASONS


TOKEN_SPLITS = ("train", "val")
METADATA_FILENAME = "metadata.json"


def token_cache_dir(artifacts_dir: Path, backbone: str, block: int, pairs_path: Path) -> Path:
    return artifacts_dir / f"tokens_{backbone}_block{block}_{Path(pairs_path).stem}"


@torch.inference_mode()
def extract_split_tokens(
    backbone: nn.Module,
    dataset: SeasoNetDataset,
    block: int,
    output_dir: Path,
    split: str,
    device: torch.device,
    image_size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    batch_size: int,
    num_workers: int,
) -> list[str]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    arrays = {}
    start = 0
    for batch in tqdm(loader, desc=f"Extracting {split} tokens", leave=False):
        for season, images in zip(SEASONS, (batch.anchor, batch.positive)):
            images = preprocess_images(images.to(device), image_size, mean, std)
            tokens = tokens_before_block(backbone, images, block).half().cpu().numpy()
            if season not in arrays:
                arrays[season] = open_memmap(
                    output_dir / f"{split}_{season}.npy",
                    mode="w+",
                    dtype=np.float16,
                    shape=(len(dataset), *tokens.shape[1:]),
                )
            arrays[season][start:start + len(tokens)] = tokens
        start += len(batch.anchor)

    for array in arrays.values():
        array.flush()
    return dataset.pairs[Pairs.LOC_ID].tolist()


def write_token_cache_metadata(output_dir: Path, metadata: dict) -> None:
    (output_dir / METADATA_FILENAME).write_text(json.dumps(metadata))


def load_token_cache(cache_dir: Path, pairs: pd.DataFrame, backbone: str, block: int) -> dict:
    metadata_path = cache_dir / METADATA_FILENAME
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Token cache not found: {cache_dir}. Run scripts/extract_tokens.py first.")

    metadata = json.loads(metadata_path.read_text())
    if metadata["backbone"] != backbone or metadata["block"] != block:
        raise ValueError(
            f"Token cache {cache_dir.name} holds {metadata['backbone']} block {metadata['block']}, "
            f"expected {backbone} block {block}."
        )

    splits = {}
    for split in TOKEN_SPLITS:
        expected = pairs.loc[pairs[Pairs.SPLIT] == split, Pairs.LOC_ID].astype(str).tolist()
        if metadata["loc_ids"][split] != expected:
            raise ValueError(f"Token cache {cache_dir.name} is stale for the {split} split. Re-run extraction.")
        splits[split] = {
            season: np.load(cache_dir / f"{split}_{season}.npy", mmap_mode="r")
            for season in SEASONS
        }
    return splits


def gather_tokens(tokens: np.ndarray, rows: torch.Tensor) -> torch.Tensor:
    # Fancy indexing copies just these rows out of the memory-mapped file.
    return torch.from_numpy(tokens[rows.numpy()]).float()
