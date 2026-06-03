#!/bin/bash
# ── MIA Grid: GPU jobs (model=mlp, all attacks, all datasets) ───────────────
# Cheaha GPU partition: pascalnodes (unlimited quota, 12h limit)
#
# Array size: 7 datasets × 3 attacks × 1 model (mlp) = 21 tasks
# Submit: sbatch slurm/mia_gpu_array.sh

#SBATCH --job-name=dpri_mia_gpu
#SBATCH --partition=pascalnodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-20
#SBATCH --output=logs/mia_gpu_%A_%a.out
#SBATCH --error=logs/mia_gpu_%A_%a.err

set -e
mkdir -p logs

PROJECT_DIR="$(dirname "$(dirname "$(realpath "$0")")")"
cd "$PROJECT_DIR"

# ── Cheaha module + conda activation ────────────────────────────────────────
# Cheaha uses Anaconda3; adjust version tag if needed (check: module avail Anaconda)
module load Anaconda3 2>/dev/null \
  || module load Miniconda3 2>/dev/null \
  || echo "WARNING: no anaconda module found — assuming conda is in PATH"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dpri

# Confirm GPU is visible
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

# ── Config mapping ───────────────────────────────────────────────────────────
DATASETS=(adult compas purchase100 texas100 nhanes movielens gowalla)
ATTACKS=(loss_threshold shadow_model lira)
MODEL=mlp   # this script is GPU-only for MLP

# idx → (dataset, attack)
# 21 tasks: 7 datasets × 3 attacks
idx=$SLURM_ARRAY_TASK_ID
ds_idx=$((idx / 3))
atk_idx=$((idx % 3))

DATASET=${DATASETS[$ds_idx]}
ATTACK=${ATTACKS[$atk_idx]}

echo "GPU Task $SLURM_ARRAY_TASK_ID → dataset=$DATASET  attack=$ATTACK  model=$MODEL"

python experiments/run_mia_grid.py \
    --dataset  "$DATASET" \
    --attack   "$ATTACK" \
    --model    "$MODEL" \
    --data_dir data/processed \
    --out_dir  results/mia_grid \
    --seed     42
