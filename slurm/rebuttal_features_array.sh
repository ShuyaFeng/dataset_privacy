#!/bin/bash
# ── Rebuttal: per-dataset geometric quantities (rebuttal_features.py) ──────
# 31 datasets = 31 tasks, CPU only. texas100 / purchase100 are the slow ones.
# Submit: sbatch slurm/rebuttal_features_array.sh
#SBATCH --job-name=dpri_rebfeat
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-30
#SBATCH --output=logs/rebfeat_%A_%a.out
#SBATCH --error=logs/rebfeat_%A_%a.err
set -e
mkdir -p logs
cd "$SLURM_SUBMIT_DIR"
module load Anaconda3 2>/dev/null || module load Miniconda3 2>/dev/null || echo "WARNING: no anaconda module found"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dpri
DATASETS=(adult compas purchase100 texas100 nhanes movielens gowalla covtype digits creditg spambase mushroom electricity letter optdigits pendigits satimage segment vehicle ionosphere phoneme bankmarketing magic nomao har gasdrift mnist fashionmnist jm1 kc1 breastw)
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}
echo "rebuttal features task $SLURM_ARRAY_TASK_ID -> $DATASET"
python experiments/rebuttal_features.py --dataset "$DATASET" --data_dir data/processed --out_dir results/rebuttal/features
