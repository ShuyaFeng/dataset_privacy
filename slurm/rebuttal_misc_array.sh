#!/bin/bash
# ── Rebuttal: encoding sensitivity (6 datasets) + benchmark recipe (9) ──────
# 15 tasks, CPU (MLP trains on CPU here; datasets are <= 30k rows).
#
# Submit:  sbatch slurm/rebuttal_misc_array.sh

#SBATCH --job-name=dpri_misc
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-14
#SBATCH --output=logs/misc_%A_%a.out
#SBATCH --error=logs/misc_%A_%a.err

set -e
mkdir -p logs
cd "$SLURM_SUBMIT_DIR"

module load Anaconda3 2>/dev/null || module load Miniconda3 2>/dev/null || echo "WARNING: no anaconda module"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dpri

ENC=(adult compas mushroom creditg bankmarketing nomao)
RECIPE=(adult covtype nomao bankmarketing letter electricity magic mnist gowalla)

idx=$SLURM_ARRAY_TASK_ID
if (( idx < ${#ENC[@]} )); then
  echo "encoding task -> ${ENC[$idx]}"
  python experiments/rebuttal_encoding.py --dataset "${ENC[$idx]}" --raw_dir data/raw
else
  j=$((idx - ${#ENC[@]}))
  echo "recipe task -> ${RECIPE[$j]}"
  python experiments/rebuttal_benchmark_recipe.py --dataset "${RECIPE[$j]}" --data_dir data/processed
fi
