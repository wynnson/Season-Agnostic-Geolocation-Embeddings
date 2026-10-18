import torch
import tifffile
import pandas as pd

from collections.abc import Callable
from torch.utils.data import Dataset
from typing import NamedTuple
from pathlib import Path

from preprocessing.normalization import normalize_rgb
from preprocessing.schema import Pairs


class SeasoNetData(NamedTuple):
    anchor: torch.Tensor
    positive: torch.Tensor
    loc_id: str


class SeasoNetDataset(Dataset):
    def __init__(
        self,
        pairs: pd.DataFrame,
        split: str,
        data_dir: Path | str,
        transform: Callable = None
    ):
        split_filter            = pairs[Pairs.SPLIT] == split
        self.pairs              = pairs[split_filter].reset_index(drop=True)
        self.data_dir           = Path(data_dir)
        self.transform          = transform

    def load_image_rgb(self, rel_path: str):
        path = self.data_dir / rel_path
        files = path.glob("*_10m_RGB.tif")
        match = next(files, None)

        if match is None:
            raise FileNotFoundError(f"No file found in {path}")

        rgb = tifffile.imread(match)    # H, W, C
        rgb = normalize_rgb(rgb)
        rgb = torch.from_numpy(rgb)
        rgb = rgb.permute(2, 0, 1)      # C, H, W
        return rgb

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index) -> SeasoNetData:
        row = self.pairs.iloc[index]
        summer_image = self.load_image_rgb(row[Pairs.PATH_SUMMER])
        snow_image = self.load_image_rgb(row[Pairs.PATH_SNOW])
        loc_id = str(row[Pairs.LOC_ID])

        if self.transform:
            summer_image = self.transform(summer_image)
            snow_image = self.transform(snow_image)

        return SeasoNetData(
            anchor=summer_image,
            positive=snow_image,
            loc_id=loc_id
        )
