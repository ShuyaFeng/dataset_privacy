#!/bin/bash
# ── Rebuttal: aggregate everything into results/rebuttal/rebuttal_summary.md ──
# Run after the grid, features, (optional) tpr/dp/subsample jobs. ~30-60 min (permutation + corpus-size refits).
# Submit: sbatch slurm/rebuttal_analysis.sh
#SBATCH --job-name=dpri_rebuttal
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --output=logs/rebuttal_%j.out
#SBATCH --error=logs/rebuttal_%j.err
set -e
mkdir -p logs
cd "$SLURM_SUBMIT_DIR"
module load Anaconda3 2>/dev/null || module load Miniconda3 2>/dev/null || echo "WARNING: no anaconda module found"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dpri
python experiments/check_ground_truth_variance.py
python experiments/run_regression.py
python experiments/run_mia_tpr_at_fpr.py --aggregate || true
python experiments/rebuttal_experiments.py --all
