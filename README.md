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

## PACE GPU

From the local repository root, set `REMOTE` to the PACE login host and destination directory. Create the destination directory first if needed.

```bash
REMOTE='username@login-host:/path/to/Season-Agnostic-Geolocation-Embeddings/'
ssh username@login-host 'mkdir -p /path/to/Season-Agnostic-Geolocation-Embeddings'
```

Sync the code, excluding local data and generated artifacts:

```bash
rsync -avh --progress \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude '__pycache__/' \
  --exclude '.env' \
  --exclude 'data/' \
  --exclude 'artifacts/' \
  ./ \
  "$REMOTE"
```

After changing the Slurm job script or training configuration, sync only the changed files:

```bash
rsync -avh scripts/train_patch_attention.sbatch \
  "$REMOTE"scripts/

rsync -avh config/train_config.yaml \
  "$REMOTE"config/
```

Sync the filtered train/validation data bundle:

```bash
rsync -avh --progress \
  artifacts/cluster_bundle/ \
  "$REMOTE"
```

Sync any cached weights. `--partial --inplace` retains partial progress when the transfer is interrupted and works with the older `rsync` shipped by macOS:

```bash
rsync -avh --progress --partial --inplace \
  artifacts/patch_features_dinov3_vits16_pairs_clouds50_block10km_buffer2000m.pt \  # replace with your cache path
  "$REMOTE"artifacts/
```

Sync the DINOv3 weights:

```bash
rsync -avh --progress \
  artifacts/weights/ \
  "$REMOTE"artifacts/weights/
```

On PACE, run this on an allocated GPU node:

```bash
module load uv/0.9.17
module load cuda/12.9.1
uv sync --locked --no-default-groups --group nvidia
uv run --no-sync python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name())"
```

Submit the patch-attention training job:

```bash
sbatch scripts/train_patch_attention.sbatch
squeue -u "$USER"
```

Follow the running job's training output and errors, replacing `<job-id>` with the ID returned by `sbatch`:

```bash
tail -f slurm-patch-attn-<job-id>.out
tail -f slurm-patch-attn-<job-id>.err
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

Train LoRA on cached DINOv3 tokens. The cache holds the tokens entering the first block in `model.lora.target_modules` (~17 GB for block 10), so only the LoRA blocks and heads run during training. `model.tuning=frozen` uses the same cache for a matched frozen baseline, and `model.patch_attention.enabled` works with either:

```bash
uv run python scripts/extract_tokens.py
uv run python scripts/train_cached_lora.py model.tuning=lora
```

Plot last-block CLS self-attention for trained checkpoints on one location, with the summer/snow attention-map correlation (example output in `figures/`):

```bash
uv run python scripts/plot_cls_attention.py <loc_id> figures/<name>.png \
  --run "Frozen=artifacts/checkpoints/<frozen>_best.pt" --run "LoRA=artifacts/checkpoints/<lora>_best.pt"
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
