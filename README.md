# Season-Agnostic Geolocation Embeddings for UAV Localization

CS 4644/7643 project by Wynnson Zheng and Erik Larson.

We adapt a pretrained DINOv2 encoder (with LoRA) to produce satellite-tile embeddings that keep geographic information and drop seasonal appearance. Training combines an InfoNCE contrastive loss over cross-season pairs of the same location with a season-adversarial classifier behind a gradient reversal layer. We evaluate with Recall@1 and Recall@5 for retrieval across seasons.

## Repository layout

```
notebooks/        Jupyter notebooks, numbered in the order they should be run
requirements.txt  Python dependencies
data              Symlink to your local SeasoNet copy (git-ignored)
```

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Data

We use the [SeasoNet](https://zenodo.org/records/6979994) dataset (Sentinel-2 tiles over Germany). Download these files from Zenodo:

| File | Size |
|---|---|
| `meta.csv` | 0.25 GB |
| `splits.zip` | 0.01 GB |
| `snow.zip` | 14.1 GB |
| `summer.zip` | 62.5 GB |

Unzipped, the data takes about 87 GB and about 5.8 million files. Keep it outside the repo and link it in:

```bash
ln -s /path/to/seasonet data
```

The folder should contain `meta.csv`, `splits/`, `summer/` and `snow/`. Notebooks also accept a `SEASONET_DIR` environment variable instead of the symlink.

Optional, macOS only: to stop Spotlight from indexing millions of tiles, either add a `.noindex` suffix to the data's parent folder or add the folder under System Settings → Spotlight → Search Privacy. The data works from any location either way.

## Notebooks

| Notebook | Purpose |
|---|---|
| `01_eda.ipynb` | Explore the Summer and Snow subsets, list cleaning decisions |

## EDA findings so far

- `meta.csv` has 1,759,830 rows across five seasons. We keep Summer (481,456 tiles, 2018) and Snow (99,458 tiles, 2019).
- 83,739 locations have both a Summer and a Snow tile. These are our contrastive pairs.
- Each tile is 120×120 px at 10 m (1.2 km per side). RGB is uint16 surface reflectance.
- Snow tiles are about 5× brighter than Summer tiles and often cloudy (median 5%, max 90% cloud). Every Snow tile has at least 10% snow cover.
- The official SeasoNet splits are random over tiles. Grid 2 is offset half a tile from Grid 1, so about 99% of test tiles overlap a training tile. We need our own geographic split.

See the decision table at the end of `notebooks/01_eda.ipynb` for open cleaning questions.
