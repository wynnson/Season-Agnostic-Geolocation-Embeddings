import pandas as pd

from pathlib import Path
from src.preprocessing.schema import Meta, Pairs


SPLIT_SUMMER = "split_summer"
SPLIT_SNOW  = "split_snow"


def load_pairs(data_dir: Path) -> pd.DataFrame:
    meta_path = data_dir / Meta.PATH_TO_METADATA

    if not meta_path.is_file():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")

    meta = pd.read_csv(meta_path)
    tiles = get_valid_tiles(meta)
    tiles = add_location_id(tiles)
    pairs = construct_pairs(tiles)

    return pairs


def get_valid_tiles(meta: pd.DataFrame) -> pd.DataFrame:
    # Filters:
    is_allowed_season                   = meta[Meta.SEASON].isin(Meta.ALLOWED_SEASONS)
    is_allowed_cloud_coverage           = meta[Meta.CLOUDS] <= Meta.MAX_CLOUD_COVER

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


def attach_splits(pairs: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    splits = []
    for split_type in ["train", "val", "test"]:
        path = data_dir / f"splits/{split_type}.csv"
        rows = pd.read_csv(path, header=None, names=[Meta.INDEX])
        rows = rows.assign(split=split_type)
        splits.append(rows)

    split_table = pd.concat(splits, ignore_index=True)

    # creates 2 tables with renamed splits 
    summer_split_table = split_table.rename(
        columns={
            Meta.INDEX: Pairs.INDEX_SUMMER,
            Pairs.SPLIT: SPLIT_SUMMER,
        }
    )

    snow_split_table = split_table.rename(
        columns={
            Meta.INDEX: Pairs.INDEX_SNOW,
            Pairs.SPLIT: SPLIT_SNOW,
        }
    )

    # merge these 2 tables on split matches
    # ex: split_snow = train, split_summer = train
    # inner join will combine them
    pairs_with_splits = pairs.merge(
        summer_split_table,
        on=Pairs.INDEX_SUMMER,
        how="inner",
    ).merge(
        snow_split_table,
        on=Pairs.INDEX_SNOW,
        how="inner"
    )

    if not (pairs_with_splits[SPLIT_SUMMER] == pairs_with_splits[SPLIT_SNOW]).all():
        raise ValueError("Summer and Snow tiles have mismatched splits")

    pairs_with_splits[Pairs.SPLIT] = pairs_with_splits[SPLIT_SUMMER]

    # cleanup by dropping useless cols
    pairs_with_splits = pairs_with_splits.drop(
        columns=[
            SPLIT_SUMMER, 
            SPLIT_SNOW
        ]
    )

    return pairs_with_splits
