#!/bin/bash
# ── DPRI feature computation: one dataset per task ──────────────────────────
# CPU only (k-NN, LOF, IsolationForest, silhouette — all sklearn CPU).
# Must run on Slurm, NOT the login node: Purchase100 (197k x 600) and
# Texas100 (67k x 6169) k-NN are too heavy for a login node.
#
# Array: 7 datasets = 7 tasks
# Submit: sbatch slurm/dpri_array.sh
# After all tasks finish: python experiments/run_dpri.py --merge

#SBATCH --job-name=dpri_feats
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-6
#SBATCH --output=logs/dpri_%A_%a.out
#SBATCH --error=logs/dpri_%A_%a.err

set -e
mkdir -p logs
cd "$SLURM_SUBMIT_DIR"

module load Anaconda3 2>/dev/null \
  || module load Miniconda3 2>/dev/null \
  || echo "WARNING: no anaconda module found — assuming conda is in PATH"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dpri

DATASETS=(adult compas purchase100 texas100 nhanes movielens gowalla)
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

echo "DPRI task $SLURM_ARRAY_TASK_ID → dataset=$DATASET"

python experiments/run_dpri.py \
    --dataset  "$DATASET" \
    --data_dir data/processed \
    --out_dir  results/dpri
