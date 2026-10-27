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

Set `evaluation.mode: "frozen"` in `config/evaluation_config.yaml` to evaluate frozen model instead.

### DINOv3

DINOv3 weights are gated. Accept the license on the [model page](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m) or through the [DINOv3 repo](https://github.com/facebookresearch/dinov3), download `dinov3_vits16_pretrain_lvd1689m-08c60483.pth`, and put it in `artifacts/weights/`. Then select it with config overrides (or the same keys in the config files):

```bash
uv run python scripts/evaluate.py evaluation.mode=frozen \
  model.repository=facebookresearch/dinov3 model.backbone=dinov3_vits16 \
  model.weights=artifacts/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth
```

`scripts/extract_features.py` and `scripts/train_cached.py` take the same overrides.

Train the model:

```bash
uv run python scripts/train.py
```

Resume interrupted raw-image training from its last checkpoint:

```bash
uv run python scripts/train.py --resume artifacts/checkpoints/<run_name>_last.pt
```

Train heads on cached frozen-DINOv2 features:

```bash
uv run python scripts/train_cached.py
```

Resume interrupted cached-feature training from its last checkpoint:

```bash
uv run python scripts/train_cached.py --resume artifacts/checkpoints/<run_name>_last.pt
```

Train LoRA on cached DINOv3 tokens. The cache holds the tokens entering the first block in `model.lora.target_modules` (~17 GB for block 10), so only the LoRA blocks and heads run during training. `model.tuning=frozen` uses the same cache for a matched frozen baseline:

```bash
uv run python scripts/extract_tokens.py
uv run python scripts/train_cached_lora.py model.layer_mixer.enabled=false model.tuning=lora
```

Use the same model architecture and scheduler configuration when resuming. To extend a completed cosine-scheduled run, start a new fine-tuning phase with an explicitly chosen scheduler rather than resuming its final checkpoint.

Checkpoints are saved as `<run_name>_best.pt` and `<run_name>_last.pt` under `artifacts/checkpoints`. Set `checkpoint.run_name` in `config/train_config.yaml` for each independent run. View training metrics with:

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
