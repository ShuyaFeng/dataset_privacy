#!/bin/bash
# ── Rebuttal: merge DPRI features, ground truth, regression, and the full aggregation ──
# Submit AFTER the grid / raw / misc / dp arrays (or with --dependency=afterany:<jobids>).
#   sbatch slurm/rebuttal_analysis.sh
#SBATCH --job-name=dpri_rebuttal
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --output=logs/rebuttal_%j.out
#SBATCH --error=logs/rebuttal_%j.err
set -u
mkdir -p logs results/regression results/rebuttal
cd "$SLURM_SUBMIT_DIR"
module load Anaconda3 2>/dev/null || module load Miniconda3 2>/dev/null || echo "WARNING: no anaconda module found"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dpri
python experiments/run_dpri.py --merge
# the paper's scripts read results/mia_grid; the rebuttal grid lives in results/mia_grid_v2
if [ ! -e results/mia_grid ] || [ -z "$(ls -A results/mia_grid 2>/dev/null)" ]; then rm -rf results/mia_grid; ln -s mia_grid_v2 results/mia_grid; fi
python experiments/check_ground_truth_variance.py
python experiments/run_regression.py
python experiments/run_mia_tpr_at_fpr.py --aggregate || true
python experiments/rebuttal_experiments.py --all --n_boot 10000 --n_perm 1000 --n_sub 200
echo "=== DONE: see results/rebuttal/rebuttal_summary.md ==="
