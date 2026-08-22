#!/bin/bash
# ── Rebuttal: one-hot variants of Adult and COMPAS through the full 9-config grid ──
# 2 datasets x 3 attacks x 3 models = 18 tasks (CPU partition; MLP on CPU is fine at this size)
# Prerequisite: python scripts/make_onehot_variants.py ; python experiments/run_dpri.py --dataset adult_onehot (and compas_onehot)
# Submit: sbatch slurm/rebuttal_grid_onehot.sh
#SBATCH --job-name=dpri_onehot
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-17
#SBATCH --output=logs/onehot_%A_%a.out
#SBATCH --error=logs/onehot_%A_%a.err
set -e
mkdir -p logs
cd "$SLURM_SUBMIT_DIR"
module load Anaconda3 2>/dev/null || module load Miniconda3 2>/dev/null || echo "WARNING: no anaconda module found"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dpri
DATASETS=(adult_onehot compas_onehot)
ATTACKS=(loss_threshold shadow_model lira)
MODELS=(mlp xgboost rf)
idx=$SLURM_ARRAY_TASK_ID
ds_idx=$((idx / 9)); rem=$((idx % 9)); atk_idx=$((rem / 3)); mdl_idx=$((rem % 3))
DATASET=${DATASETS[$ds_idx]}; ATTACK=${ATTACKS[$atk_idx]}; MODEL=${MODELS[$mdl_idx]}
echo "one-hot task $idx -> $DATASET $ATTACK $MODEL"
python experiments/run_mia_grid.py --dataset "$DATASET" --attack "$ATTACK" --model "$MODEL" --data_dir data/processed --out_dir results/mia_grid --seed 42
