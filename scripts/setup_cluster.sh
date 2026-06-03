#!/bin/bash
# Run this script ONCE on the cluster login node.
# Usage: bash scripts/setup_cluster.sh

set -e

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
echo "Project root: $PROJECT_DIR"

# ── 1. Load conda (adjust module name to your cluster) ──────────────────────
# Common names: anaconda3, miniconda3, conda
module load anaconda3 2>/dev/null || module load miniconda3 2>/dev/null || true

if ! command -v conda &>/dev/null; then
    echo "ERROR: conda not found. Ask your sysadmin for the correct module name."
    exit 1
fi

# ── 2. Create environment ────────────────────────────────────────────────────
echo "Creating conda environment 'dpri' ..."
conda env create -f "$PROJECT_DIR/environment.yml" --force
echo "Environment created."

# ── 3. Activate and verify ───────────────────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dpri

python - <<'EOF'
import sklearn, xgboost, torch, numpy, pandas
print(f"sklearn  {sklearn.__version__}")
print(f"xgboost  {xgboost.__version__}")
print(f"torch    {torch.__version__}")
print(f"numpy    {numpy.__version__}")
print(f"pandas   {pandas.__version__}")
print("All packages OK.")
EOF

# ── 4. Download datasets (runs on login node, no GPU needed) ─────────────────
echo "Downloading datasets ..."
python "$PROJECT_DIR/scripts/download_data.py" \
    --data_dir "$PROJECT_DIR/data/raw" \
    --out_dir  "$PROJECT_DIR/data/processed"

echo ""
echo "Setup complete."
echo "Next: manually place purchase100.npz and texas100.npz in data/raw/"
echo "Then run: python scripts/download_data.py   (it will process them)"
