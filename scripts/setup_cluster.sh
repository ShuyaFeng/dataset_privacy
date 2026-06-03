#!/bin/bash
# Step 3: Quick setup — create output directories and print next steps.
# Run after install_env.sh and download_data.sh. Takes <5 seconds.
# Usage: bash scripts/setup_cluster.sh

set -e

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$PROJECT_DIR/results/mia_grid"
mkdir -p "$PROJECT_DIR/results/dpri"
mkdir -p "$PROJECT_DIR/results/regression"

echo "Directories created."
echo ""
echo "Ready to submit jobs:"
echo "  sbatch slurm/mia_gpu_array.sh   # MLP  — 21 tasks on pascalnodes"
echo "  sbatch slurm/mia_cpu_array.sh   # XGB+RF — 42 tasks on short"
echo ""
echo "Monitor: squeue -u \$USER"
