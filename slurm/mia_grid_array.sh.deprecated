#!/bin/bash
#SBATCH --job-name=dpri_mia_grid
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-62          # 63 valid configs (see note below)
#SBATCH --output=logs/mia_grid_%A_%a.out
#SBATCH --error=logs/mia_grid_%A_%a.err

# ── Note on array size ──────────────────────────────────────────────────────
# 7 datasets × 3 attacks × 3 models = 63 configs
# (purchase100 and texas100 are included but will self-skip if .npz not present)
# ───────────────────────────────────────────────────────────────────────────

set -e

# Activate environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dpri

PROJECT_DIR="$(dirname "$(dirname "$(realpath "$0")")")"
cd "$PROJECT_DIR"
mkdir -p logs

# Build the full config list
DATASETS=(adult compas purchase100 texas100 nhanes movielens gowalla)
ATTACKS=(loss_threshold shadow_model lira)
MODELS=(mlp xgboost rf)

# Map SLURM_ARRAY_TASK_ID → (dataset, attack, model)
idx=$SLURM_ARRAY_TASK_ID
n_attacks=${#ATTACKS[@]}     # 3
n_models=${#MODELS[@]}       # 3
n_per_dataset=$((n_attacks * n_models))  # 9

ds_idx=$((idx / n_per_dataset))
remainder=$((idx % n_per_dataset))
atk_idx=$((remainder / n_models))
mdl_idx=$((remainder % n_models))

DATASET=${DATASETS[$ds_idx]}
ATTACK=${ATTACKS[$atk_idx]}
MODEL=${MODELS[$mdl_idx]}

echo "Task $SLURM_ARRAY_TASK_ID → dataset=$DATASET attack=$ATTACK model=$MODEL"

python experiments/run_mia_grid.py \
    --dataset  "$DATASET" \
    --attack   "$ATTACK" \
    --model    "$MODEL" \
    --data_dir data/processed \
    --out_dir  results/mia_grid \
    --seed     42
