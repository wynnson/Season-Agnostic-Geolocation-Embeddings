#!/bin/bash
#SBATCH --job-name=extract-tokens
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

uv run --no-sync python scripts/extract_tokens.py model.tuning=lora
