#!/bin/bash
# ── Rebuttal: subsampling sweep (Corollary 1), single GPU job ──
# Submit: sbatch slurm/rebuttal_subsample.sh
#SBATCH --job-name=dpri_subsample
#SBATCH --partition=pascalnodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/subsample_%j.out
#SBATCH --error=logs/subsample_%j.err
set -e
mkdir -p logs
cd "$SLURM_SUBMIT_DIR"
module load Anaconda3 2>/dev/null || module load Miniconda3 2>/dev/null || echo "WARNING: no anaconda module found"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dpri
python experiments/rebuttal_subsample_sweep.py --datasets adult,covtype,mnist,purchase100 --fractions 1,0.5,0.25,0.125,0.0625 --seeds 0,1,2 --models mlp,xgboost
