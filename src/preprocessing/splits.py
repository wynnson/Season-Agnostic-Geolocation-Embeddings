import hashlib

import numpy as np
import pandas as pd

from scipy.spatial import KDTree

from src.preprocessing.schema import Pairs
from src.preprocessing.spatial import EARTH_RADIUS_M, lon_lat_to_unit_sphere


SPLIT_FRACTIONS         = {"train": 0.75, "val": 0.1, "test": 0.15}
SPLIT_PRIORITY          = {"train": 0, "val": 1, "test": 2}
REFERENCE_LATITUDE      = 51.0  # centre of Germany; sets block width in degrees
METERS_PER_DEGREE       = np.deg2rad(1) * EARTH_RADIUS_M

BLOCK_X                 = "block_x"
BLOCK_Y                 = "block_y"


def assign_geographic_splits(
    pairs: pd.DataFrame,
    block_size_m: float = 10_000,
    buffer_m: float = 2_000,
    fractions: dict[str, float] = SPLIT_FRACTIONS,
    seed: int = 42,
) -> pd.DataFrame:
    pairs = add_block_ids(pairs, block_size_m)
    pairs[Pairs.SPLIT] = assign_blocks(pairs, fractions, seed)
    return drop_cross_split_neighbors(pairs, buffer_m)


def add_block_ids(pairs: pd.DataFrame, block_size_m: float) -> pd.DataFrame:
    block_lat = block_size_m / METERS_PER_DEGREE
    block_lon = block_lat / np.cos(np.deg2rad(REFERENCE_LATITUDE))

    pairs = pairs.copy()
    pairs[BLOCK_X] = np.floor(pairs[Pairs.LONGITUDE_SUMMER] / block_lon).astype(int)
    pairs[BLOCK_Y] = np.floor(pairs[Pairs.LATITUDE_SUMMER] / block_lat).astype(int)
    return pairs


def assign_blocks(pairs: pd.DataFrame, fractions: dict[str, float], seed: int) -> pd.Series:
    # Hashing each block id (instead of shuffling) keeps a block's split stable
    # when filters such as the cloud threshold add or remove other blocks.
    blocks = pairs[[BLOCK_X, BLOCK_Y]].drop_duplicates()
    blocks["u"] = [
        block_uniform(seed, bx, by) for bx, by in zip(blocks[BLOCK_X], blocks[BLOCK_Y])
    ]

    names = list(fractions)
    edges = np.cumsum([fractions[n] for n in names])
    blocks[Pairs.SPLIT] = [names[i] for i in np.searchsorted(edges, blocks["u"], side="right")]

    merged = pairs[[BLOCK_X, BLOCK_Y]].merge(
        blocks[[BLOCK_X, BLOCK_Y, Pairs.SPLIT]], on=[BLOCK_X, BLOCK_Y], how="left"
    )
    return merged[Pairs.SPLIT].to_numpy()


def block_uniform(seed: int, bx: int, by: int) -> float:
    digest = hashlib.sha256(f"{seed}:{bx}:{by}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2 ** 64


def drop_cross_split_neighbors(pairs: pd.DataFrame, buffer_m: float) -> pd.DataFrame:
    """Drop tiles within buffer_m of a tile in a higher-priority split (test > val > train).

    Tiles are 1.2 km wide and Grid 2 is offset half a tile from Grid 1, so
    overlapping tiles are at most ~1.7 km apart. A 2 km buffer removes all overlap.
    """
    xyz = lon_lat_to_unit_sphere(
        pairs[Pairs.LONGITUDE_SUMMER].to_numpy(), pairs[Pairs.LATITUDE_SUMMER].to_numpy()
    )
    priority = pairs[Pairs.SPLIT].map(SPLIT_PRIORITY).to_numpy()
    neighbors = KDTree(xyz).query_ball_point(xyz, r=buffer_m / EARTH_RADIUS_M)

    keep = np.array([
        priority[i] >= priority[idx].max() for i, idx in enumerate(neighbors)
    ])
    return pairs[keep].reset_index(drop=True)
