import pandas as pd

from pathlib import Path
from src.preprocessing.schema import Meta, Pairs


def load_pairs(data_dir: Path, max_cloud_cover: float) -> pd.DataFrame:
    meta_path = data_dir / Meta.PATH_TO_METADATA

    if not meta_path.is_file():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")

    meta = pd.read_csv(meta_path)
    tiles = get_valid_tiles(meta, max_cloud_cover)
    tiles = add_location_id(tiles)
    pairs = construct_pairs(tiles)

    return pairs


def get_valid_tiles(meta: pd.DataFrame, max_cloud_cover: float) -> pd.DataFrame:
    # Filters:
    is_allowed_season                   = meta[Meta.SEASON].isin(Meta.ALLOWED_SEASONS)
    is_allowed_cloud_coverage           = meta[Meta.CLOUDS] <= max_cloud_cover

    tiles = meta[
        is_allowed_season &
        is_allowed_cloud_coverage
    ].copy()

    return tiles


def add_location_id(tiles: pd.DataFrame) -> pd.DataFrame:
    tiles[Pairs.LOC_ID] = (
        tiles[Meta.GRID].astype(str) + "_"    # grid 1 / 2 (half-tile offsets of grid 1)
        + tiles[Meta.LONGITUDE].map("{:.6f}".format) + "_"
        + tiles[Meta.LATITUDE].map("{:.6f}".format)
    )
    
    return tiles


def construct_pairs(tiles: pd.DataFrame) -> pd.DataFrame: 
    is_summer       = tiles[Meta.SEASON] == Meta.SUMMER
    is_snow         = tiles[Meta.SEASON] == Meta.SNOW

    summer          = tiles[is_summer]
    snow            = tiles[is_snow]

    pairs = summer.merge(
        snow,
        on=Pairs.LOC_ID,
        suffixes=("_summer", "_snow"),
    )

    return pairs
