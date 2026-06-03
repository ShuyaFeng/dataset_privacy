#!/bin/bash
# Step 2: Download and preprocess all datasets.
# Run ONCE on login node after install_env.sh. Takes ~5-15 min.
# Usage: bash scripts/download_data.sh

set -e

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

module load Anaconda3 2>/dev/null || module load Miniconda3 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dpri

python "$PROJECT_DIR/scripts/download_data.py" \
    --data_dir "$PROJECT_DIR/data/raw" \
    --out_dir  "$PROJECT_DIR/data/processed"

echo ""
echo "Done. Check data/processed/ for .npz files."
echo ""
echo "NOTE: Purchase100 and Texas100 need manual download:"
echo "  https://github.com/privacytrustlab/datasets"
echo "  Place .npz files in data/raw/, then re-run this script."
