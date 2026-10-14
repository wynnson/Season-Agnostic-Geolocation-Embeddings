# Season Agnostic Geolocation Embeddings

We train foundation models to represent the same place (aerial view), different times as similar embeddings. We use contrastive learning with adversarial training to push similar locations closer despite seasonal differences which helps the model understand the place not season. We call these embeddings Season Agnostic Geolocation Embeddings (SAGE).

## Repository layout

```
notebooks/              Jupyter notebooks, numbered in the order they should be run
src/preprocessing/      Reusable data cleaning, normalization, and spatial helpers
artifacts/              Generated preprocessing outputs (git-ignored)
pyproject.toml          Python dependencies
data                    Symlink to your local SeasoNet copy (git-ignored)
```

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Data

We use the [SeasoNet](https://zenodo.org/records/6979994) dataset (Sentinel-2 tiles over Germany). Download these files from Zenodo:

| File | Size |
|---|---|
| `meta.csv` | 0.25 GB |
| `splits.zip` | 0.01 GB |
| `snow.zip` | 14.1 GB |
| `summer.zip` | 62.5 GB |

Unzipped, the data takes about 87 GB and about 5.8 million files. Drag it in /data folder or keep it outside the repo and link it in:

```bash
ln -s /path/to/seasonet data
```
