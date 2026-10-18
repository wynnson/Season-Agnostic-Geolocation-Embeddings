# Season Agnostic Geolocation Embeddings

We train foundation models to represent the same place (aerial view), different times as similar embeddings. We use contrastive learning with adversarial training to push similar locations closer despite seasonal differences which helps the model understand the place not season. We call these embeddings Season Agnostic Geolocation Embeddings (SAGE).

## Repository layout

```
notebooks/              Jupyter notebooks, numbered in the order they should be run
src/preprocessing/      Reusable data cleaning, normalization, and spatial helpers
src/models/             Model loading and image-to-embedding helpers
src/training/           Retrieval metrics and model-agnostic evaluation
scripts/                Training and evaluation entry points
config/                 Training, evaluation, and preprocessing configuration
artifacts/              Generated preprocessing outputs (git-ignored)
pyproject.toml          Python dependencies
data                    Symlink to your local SeasoNet copy (git-ignored)
```

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Scripts

Create geographic split, hard-negative, and manifest artifacts:

```bash
uv run python scripts/preprocess.py
```

Evaluate the configured trained checkpoint on the geographic validation split:

```bash
uv run python scripts/evaluate.py
```

Set `evaluation.mode: "frozen"` in `config/evaluation_config.yaml` to evaluate frozen DINOv2 instead.

Train the model:

```bash
uv run python scripts/train.py
```

Resume from a checkpoint:

```bash
uv run python scripts/train.py --resume artifacts/checkpoints/dinov2_vits14_geo_grl_last.pt
```

Checkpoints are saved as `<run_name>_best.pt`, `<run_name>_last.pt`, and periodic `<run_name>_epoch_XXX.pt` files under `artifacts/checkpoints`. Set `checkpoint.run_name` in `config/train_config.yaml` for each independent run. View training metrics with:

```bash
uv run tensorboard --logdir artifacts/tensorboard
```

## Data

We use the [SeasoNet](https://zenodo.org/records/6979994) dataset (Sentinel-2 tiles over Germany). Download these files from Zenodo:

| File | Size |
|---|---|
| `meta.csv` | 0.25 GB |
| `splits.zip` | 0.01 GB |
| `snow.zip` | 14.1 GB |
| `summer.zip` | 62.5 GB |

Unzipped, the data takes about 87 GB and about 5.8 million files. Keep it outside the repo and link it in. A symlink is instant, while copying 5.8 million files into the repo is very slow:

```bash
ln -s /path/to/seasonet data
```
