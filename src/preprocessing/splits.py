import hashlib

import numpy as np
import pandas as pd

from scipy.spatial import KDTree

from src.preprocessing.schema import Pairs
from src.preprocessing.spatial import EARTH_RADIUS_M, lon_lat_to_unit_sphere


METERS_PER_DEGREE       = np.deg2rad(1) * EARTH_RADIUS_M

BLOCK_X                 = "block_x"
BLOCK_Y                 = "block_y"


def assign_geographic_splits(
    pairs: pd.DataFrame,
    block_size_m: float,
    buffer_m: float,
    reference_latitude: float,
    fractions: dict[str, float],
    priority: dict[str, int],
    seed: int,
) -> pd.DataFrame:
    pairs = add_block_ids(pairs, block_size_m, reference_latitude)
    pairs[Pairs.SPLIT] = assign_blocks(pairs, fractions, seed)
    pairs = drop_cross_split_neighbors(pairs, buffer_m, priority)
    return pairs


def add_block_ids(
    pairs: pd.DataFrame,
    block_size_m: float,
    reference_latitude: float,
) -> pd.DataFrame:
    meters_per_lon_degree = np.cos(np.deg2rad(reference_latitude))
    block_lat = block_size_m / METERS_PER_DEGREE
    block_lon = block_lat / meters_per_lon_degree

    pairs = pairs.copy()
    pairs[BLOCK_X] = np.floor(pairs[Pairs.LONGITUDE_SUMMER] / block_lon).astype(int)
    pairs[BLOCK_Y] = np.floor(pairs[Pairs.LATITUDE_SUMMER] / block_lat).astype(int)
    return pairs


def assign_blocks(pairs: pd.DataFrame, fractions: dict[str, float], seed: int) -> pd.Series:
    # Hashing each block id (instead of shuffling) keeps a block's split stable
    # when filters such as the cloud threshold add or remove other blocks.
    blocks = pairs[[BLOCK_X, BLOCK_Y]].drop_duplicates().copy()

    blocks["split_score"] = [
        block_uniform(seed, x, y) 
        for x, y in zip(blocks[BLOCK_X], blocks[BLOCK_Y])
    ]

    split_names = list(fractions.keys())
    split_thresholds = np.cumsum(list(fractions.values()))
    split_indices = np.searchsorted(
        split_thresholds,
        blocks["split_score"],
        side="right"
    )

    blocks[Pairs.SPLIT] = [
        split_names[index]
        for index in split_indices
    ]

    result = pairs[[BLOCK_X, BLOCK_Y]].merge(
        blocks[[BLOCK_X, BLOCK_Y, Pairs.SPLIT]], 
        on=[BLOCK_X, BLOCK_Y], 
        how="left",
        sort=False
    )

    return pd.Series(
        result[Pairs.SPLIT].to_numpy(),
        index=pairs.index,
        name=Pairs.SPLIT,
    )


def block_uniform(seed: int, x: int, y: int) -> float:
    digest = hashlib.sha256(f"{seed}:{x}:{y}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2 ** 64


def drop_cross_split_neighbors(
    pairs: pd.DataFrame,
    buffer_m: float,
    priority: dict[str, int],
) -> pd.DataFrame:
    """Drop tiles within buffer_m of a tile in a higher-priority split (test > val > train).

    Tiles are 1.2 km wide and Grid 2 is offset half a tile from Grid 1, so
    overlapping tiles are at most ~1.7 km apart. A 2 km buffer removes all overlap.
    """
    coordinates = lon_lat_to_unit_sphere(
        pairs[Pairs.LONGITUDE_SUMMER].to_numpy(),
        pairs[Pairs.LATITUDE_SUMMER].to_numpy(),
    )

    split_priorities = pairs[Pairs.SPLIT].map(priority).to_numpy()
    radius = 2 * np.sin(buffer_m / (2 * EARTH_RADIUS_M))
    
    tree = KDTree(coordinates)
    neighbors = tree.query_ball_point(coordinates, r=radius)

    keep = []

    for index, neighbor_indices in enumerate(neighbors):
        current_priority = split_priorities[index]
        highest_priority = split_priorities[neighbor_indices].max()

        keep.append(current_priority >= highest_priority)

    return pairs.loc[keep].reset_index(drop=True)
