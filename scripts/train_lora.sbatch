#!/bin/bash
#SBATCH --job-name=lora
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

cd "$HOME/scratch/Season-Agnostic-Geolocation-Embeddings"

module load uv/0.9.17
module load cuda/12.9.1

uv run --no-sync python scripts/train_cached_lora.py \
  model.tuning=lora \
  model.patch_attention.enabled=false \
  training.batch_size=256 \
  training.sampled_hard_negatives_per_anchor=1 \
  checkpoint.run_name=dinov3_vits16_lora_residual_hard1_b256
