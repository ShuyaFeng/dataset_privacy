#!/bin/bash
# ── Rebuttal: LiRA with TPR@low FPR, all datasets × 3 models ─────────
# 31 datasets × 3 models = 93 tasks
# Submit: sbatch slurm/rebuttal_tpr_array.sh

#SBATCH --job-name=dpri_tpr
#SBATCH --partition=pascalnodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-92
#SBATCH --output=logs/tpr_%A_%a.out
#SBATCH --error=logs/tpr_%A_%a.err

set -e
mkdir -p logs

cd "$SLURM_SUBMIT_DIR"

module load Anaconda3 2>/dev/null \
  || module load Miniconda3 2>/dev/null \
  || echo "WARNING: no anaconda module found"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dpri

DATASETS=(adult compas purchase100 texas100 nhanes movielens gowalla covtype digits creditg spambase mushroom electricity letter optdigits pendigits satimage segment vehicle ionosphere phoneme bankmarketing magic nomao har gasdrift mnist fashionmnist jm1 kc1 breastw)
MODELS=(mlp xgboost rf)

idx=$SLURM_ARRAY_TASK_ID
ds_idx=$((idx / 3))
model_idx=$((idx % 3))

DATASET=${DATASETS[$ds_idx]}
MODEL=${MODELS[$model_idx]}

echo "High-res LiRA task $SLURM_ARRAY_TASK_ID -> dataset=$DATASET model=$MODEL (eval_n=2000)"

python experiments/run_mia_tpr_at_fpr.py \
    --dataset "$DATASET" \
    --model   "$MODEL" \
    --data_dir data/processed \
    --seed 42
