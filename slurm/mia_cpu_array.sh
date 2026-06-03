#!/bin/bash
# ── MIA Grid: CPU jobs (model=xgboost or rf, all attacks, all datasets) ─────
# Cheaha CPU partition: short (12h, 44 nodes per researcher)
#
# Array size: 7 datasets × 3 attacks × 2 models (xgboost, rf) = 42 tasks
# Submit: sbatch slurm/mia_cpu_array.sh

#SBATCH --job-name=dpri_mia_cpu
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-41
#SBATCH --output=logs/mia_cpu_%A_%a.out
#SBATCH --error=logs/mia_cpu_%A_%a.err

set -e
mkdir -p logs

cd "$SLURM_SUBMIT_DIR"

# ── Cheaha module + conda activation ────────────────────────────────────────
module load Anaconda3 2>/dev/null \
  || module load Miniconda3 2>/dev/null \
  || echo "WARNING: no anaconda module found — assuming conda is in PATH"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dpri

# ── Config mapping ───────────────────────────────────────────────────────────
DATASETS=(adult compas purchase100 texas100 nhanes movielens gowalla)
ATTACKS=(loss_threshold shadow_model lira)
MODELS=(xgboost rf)

# 42 tasks: 7 datasets × 3 attacks × 2 models
# idx → dataset, attack, model
idx=$SLURM_ARRAY_TASK_ID
n_attacks=3
n_models=2
n_per_dataset=$((n_attacks * n_models))   # 6

ds_idx=$((idx / n_per_dataset))
remainder=$((idx % n_per_dataset))
atk_idx=$((remainder / n_models))
mdl_idx=$((remainder % n_models))

DATASET=${DATASETS[$ds_idx]}
ATTACK=${ATTACKS[$atk_idx]}
MODEL=${MODELS[$mdl_idx]}

echo "CPU Task $SLURM_ARRAY_TASK_ID → dataset=$DATASET  attack=$ATTACK  model=$MODEL"

python experiments/run_mia_grid.py \
    --dataset  "$DATASET" \
    --attack   "$ATTACK" \
    --model    "$MODEL" \
    --data_dir data/processed \
    --out_dir  results/mia_grid \
    --seed     42
