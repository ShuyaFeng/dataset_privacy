#!/bin/bash
# One-shot setup script for UAB Cheaha HPC.
# Run ONCE on a login node (no GPU needed for setup).
# Usage: bash scripts/setup_cluster.sh

set -e

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
echo "Project root: $PROJECT_DIR"

# ── 1. Load Anaconda (Cheaha-specific) ──────────────────────────────────────
# Check what's available with: module avail Anaconda
echo "Loading Anaconda module ..."
module load Anaconda3 2>/dev/null \
  || module load Miniconda3 2>/dev/null \
  || { echo "ERROR: Cannot find Anaconda3 or Miniconda3 module."; echo "Run: module avail 2>&1 | grep -i conda"; exit 1; }

# ── 2. Create conda environment ──────────────────────────────────────────────
echo "Creating conda environment 'dpri' (this takes ~5 min) ..."

# Install PyTorch with CUDA 11.8 — matches Cheaha's available CUDA versions
conda env create -f "$PROJECT_DIR/environment.yml" --force

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dpri

# Install GPU-enabled PyTorch separately (conda-forge version may be CPU-only)
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu118 --quiet

# ── 3. Verify GPU stack ──────────────────────────────────────────────────────
echo ""
echo "Verifying Python environment ..."
python - <<'EOF'
import sklearn, xgboost, torch, numpy, pandas
print(f"  numpy    {numpy.__version__}")
print(f"  pandas   {pandas.__version__}")
print(f"  sklearn  {sklearn.__version__}")
print(f"  xgboost  {xgboost.__version__}")
print(f"  torch    {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
else:
    print("  NOTE: CUDA not visible on login node — this is normal.")
    print("        GPU will be available inside Slurm GPU jobs.")
EOF

# ── 4. Download datasets ─────────────────────────────────────────────────────
echo ""
echo "Downloading datasets (runs on login node, no GPU needed) ..."
python "$PROJECT_DIR/scripts/download_data.py" \
    --data_dir "$PROJECT_DIR/data/raw" \
    --out_dir  "$PROJECT_DIR/data/processed"

# ── 5. Create output directories ─────────────────────────────────────────────
mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$PROJECT_DIR/results/mia_grid"
mkdir -p "$PROJECT_DIR/results/dpri"
mkdir -p "$PROJECT_DIR/results/regression"

echo ""
echo "=================================================="
echo "Setup complete."
echo ""
echo "NEXT STEPS:"
echo "  1. Manually download Purchase100 + Texas100:"
echo "     https://github.com/privacytrustlab/datasets"
echo "     Place .npz files in: $PROJECT_DIR/data/raw/"
echo "     Then re-run: python scripts/download_data.py"
echo ""
echo "  2. Submit GPU jobs (MLP model):"
echo "     sbatch slurm/mia_gpu_array.sh"
echo ""
echo "  3. Submit CPU jobs (XGBoost + RF):"
echo "     sbatch slurm/mia_cpu_array.sh"
echo ""
echo "  4. Monitor:"
echo "     squeue -u \$USER"
echo "=================================================="
