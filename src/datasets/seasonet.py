import torch
import tifffile
import pandas as pd

from collections.abc import Callable
from functools import lru_cache
from torch.utils.data import Dataset
from typing import NamedTuple
from pathlib import Path

from src.preprocessing.normalization import normalize_rgb
from src.preprocessing.schema import HardNegatives, Pairs


@lru_cache(maxsize=32_768)
def resolve_rgb_path(data_dir: str, rel_path: str) -> Path:
    directory = Path(data_dir) / rel_path
    match = next(directory.glob("*_10m_RGB.tif"), None)

    if match is None:
        raise FileNotFoundError(f"No RGB TIFF found in {directory}")

    return match


class SeasoNetData(NamedTuple):
    anchor: torch.Tensor
    positive: torch.Tensor
    summer_hard_negatives: torch.Tensor
    snow_hard_negatives: torch.Tensor
    loc_id: str


class SeasoNetDataset(Dataset):
    def __init__(
        self,
        pairs: pd.DataFrame,
        split: str,
        data_dir: Path | str,
        transform: Callable | None = None,
        hard_negatives: pd.DataFrame | None = None,
        sampled_hard_negatives_per_anchor: int = 0,
    ):
        if sampled_hard_negatives_per_anchor < 0:
            raise ValueError("sampled_hard_negatives_per_anchor must be non-negative")

        split_filter = pairs[Pairs.SPLIT] == split
        self.pairs = pairs[split_filter].reset_index(drop=True).copy()
        self.pairs[Pairs.LOC_ID] = self.pairs[Pairs.LOC_ID].astype(str)
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.sampled_hard_negatives_per_anchor = sampled_hard_negatives_per_anchor

        if self.pairs[Pairs.LOC_ID].duplicated().any():
            raise ValueError("Each location id must appear once in a dataset split")

        self.pairs_by_loc_id = self.pairs.set_index(Pairs.LOC_ID)
        self.negative_ids_by_anchor = self._build_hard_negative_lookup(hard_negatives)

    def _build_hard_negative_lookup(
        self,
        hard_negatives: pd.DataFrame | None,
    ) -> dict[str, list[str]]:
        if self.sampled_hard_negatives_per_anchor == 0:
            return {}
        if hard_negatives is None:
            raise ValueError(
                "hard_negatives are required when "
                "sampled_hard_negatives_per_anchor is positive"
            )

        hard_negatives                                  = hard_negatives.copy()
        hard_negatives[HardNegatives.ANCHOR_LOC_ID]     = hard_negatives[HardNegatives.ANCHOR_LOC_ID].astype(str)
        hard_negatives[HardNegatives.NEGATIVE_LOC_ID]   = hard_negatives[HardNegatives.NEGATIVE_LOC_ID].astype(str)
        loc_ids                                         = set(self.pairs[Pairs.LOC_ID])

        valid_negatives = hard_negatives[
            hard_negatives[HardNegatives.ANCHOR_LOC_ID].isin(loc_ids)
            & hard_negatives[HardNegatives.NEGATIVE_LOC_ID].isin(loc_ids)
        ].sort_values(HardNegatives.DISTANCE_M)

        lookup = (
            valid_negatives
            .groupby(HardNegatives.ANCHOR_LOC_ID)[HardNegatives.NEGATIVE_LOC_ID]
            .agg(list)
            .to_dict()
        )

        missing = [
            loc_id
            for loc_id in self.pairs[Pairs.LOC_ID]
            if len(lookup.get(loc_id, [])) < self.sampled_hard_negatives_per_anchor
        ]

        if missing:
            raise ValueError(
                f"{len(missing)} locations have fewer than "
                f"{self.sampled_hard_negatives_per_anchor} hard negatives"
            )

        return lookup

    def load_image_rgb(self, rel_path: str):
        path = resolve_rgb_path(str(self.data_dir), rel_path)
        rgb = tifffile.imread(path)     # H, W, C
        rgb = normalize_rgb(rgb)
        rgb = torch.from_numpy(rgb)
        rgb = rgb.permute(2, 0, 1)      # C, H, W
        return rgb

    def load_pair(self, row: pd.Series) -> tuple[torch.Tensor, torch.Tensor]:
        summer_image = self.load_image_rgb(row[Pairs.PATH_SUMMER])
        snow_image = self.load_image_rgb(row[Pairs.PATH_SNOW])

        if self.transform:
            summer_image = self.transform(summer_image)
            snow_image = self.transform(snow_image)

        return summer_image, snow_image

    def load_hard_negative_pairs(
        self,
        loc_id: str,
        image_shape: torch.Size,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.sampled_hard_negatives_per_anchor == 0:
            empty = torch.empty((0, *image_shape), dtype=torch.float32,)
            return empty, empty

        candidate_ids = self.negative_ids_by_anchor[loc_id]
        selected_ids = candidate_ids[:self.sampled_hard_negatives_per_anchor]
        negative_rows = self.pairs_by_loc_id.loc[selected_ids]

        negative_pairs = [
            self.load_pair(row)
            for _, row in negative_rows.iterrows()
        ]

        summer_images = []
        snow_images = []

        for summer, snow in negative_pairs:
            summer_images.append(summer)
            snow_images.append(snow)

        summer_hard_negatives = torch.stack(summer_images)
        snow_hard_negatives = torch.stack(snow_images)

        return summer_hard_negatives, snow_hard_negatives

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index) -> SeasoNetData:
        row = self.pairs.iloc[index]
        loc_id = str(row[Pairs.LOC_ID])
        summer_image, snow_image = self.load_pair(row)
        summer_hard_negatives, snow_hard_negatives = self.load_hard_negative_pairs(
            loc_id,
            summer_image.shape,
        )

        return SeasoNetData(
            anchor=summer_image,
            positive=snow_image,
            summer_hard_negatives=summer_hard_negatives,
            snow_hard_negatives=snow_hard_negatives,
            loc_id=loc_id,
        )
