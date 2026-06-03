#!/bin/bash
# Step 1: Create conda environment + install PyTorch with CUDA.
# Run ONCE on login node. Takes ~5-10 min.
# Usage: bash scripts/install_env.sh

set -e

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

module load Anaconda3 2>/dev/null \
  || module load Miniconda3 2>/dev/null \
  || { echo "ERROR: no Anaconda module found. Run: module avail 2>&1 | grep -i conda"; exit 1; }

source "$(conda info --base)/etc/profile.d/conda.sh"

echo "Creating conda environment 'dpri' ..."
conda env create -f "$PROJECT_DIR/environment.yml" --force

conda activate dpri

echo "Installing PyTorch with CUDA 11.8 ..."
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu118 --quiet

echo ""
echo "Verifying ..."
python - <<'EOF'
import sklearn, xgboost, torch, numpy, pandas
print(f"  numpy    {numpy.__version__}")
print(f"  sklearn  {sklearn.__version__}")
print(f"  xgboost  {xgboost.__version__}")
print(f"  torch    {torch.__version__}  (CUDA: {torch.cuda.is_available()})")
print("All packages OK.")
EOF

echo ""
echo "Done. Next: bash scripts/download_data.sh"
