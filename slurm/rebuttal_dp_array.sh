#!/bin/bash
# ── Rebuttal: DP-SGD robustness (rebuttal_dp.py), one dataset per GPU task ──
# Requires: pip install opacus   (inside the dpri env)
# Submit: sbatch slurm/rebuttal_dp_array.sh        (add --lira inside for DP-shadow LiRA; ~8x slower)
#SBATCH --job-name=dpri_dp
#SBATCH --partition=pascalnodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-30
#SBATCH --output=logs/dp_%A_%a.out
#SBATCH --error=logs/dp_%A_%a.err
set -e
mkdir -p logs
cd "$SLURM_SUBMIT_DIR"
module load Anaconda3 2>/dev/null || module load Miniconda3 2>/dev/null || echo "WARNING: no anaconda module found"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dpri
DATASETS=(adult compas purchase100 texas100 nhanes movielens gowalla covtype digits creditg spambase mushroom electricity letter optdigits pendigits satimage segment vehicle ionosphere phoneme bankmarketing magic nomao har gasdrift mnist fashionmnist jm1 kc1 breastw)
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}
echo "DP task $SLURM_ARRAY_TASK_ID -> $DATASET"
python experiments/rebuttal_dp.py --dataset "$DATASET" --epsilons 1,4,8 --epochs 30 --seed 42
