#!/bin/bash
# ── Rebuttal grid v2, CPU part: model in {xgboost, rf}, all attacks ─────────
# 31 datasets x 3 attacks x 2 models = 186 tasks.
#
# Submit:            sbatch slurm/rebuttal_grid_v2_cpu.sh
# Extra seeds (loss-threshold only):
#                    sbatch --export=ALL,SEED=43,ATTACK_FILTER=loss_threshold slurm/rebuttal_grid_v2_cpu.sh
#                    sbatch --export=ALL,SEED=44,ATTACK_FILTER=loss_threshold slurm/rebuttal_grid_v2_cpu.sh

#SBATCH --job-name=dpri_v2_cpu
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-185
#SBATCH --output=logs/v2_cpu_%A_%a.out
#SBATCH --error=logs/v2_cpu_%A_%a.err

set -e
mkdir -p logs
cd "$SLURM_SUBMIT_DIR"

module load Anaconda3 2>/dev/null || module load Miniconda3 2>/dev/null || echo "WARNING: no anaconda module"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dpri

SEED=${SEED:-42}
OUT_DIR=${OUT_DIR:-results/mia_grid_v2}
ATTACK_FILTER=${ATTACK_FILTER:-}

DATASETS=(adult compas purchase100 texas100 nhanes movielens gowalla covtype digits creditg spambase mushroom electricity letter optdigits pendigits satimage segment vehicle ionosphere phoneme bankmarketing magic nomao har gasdrift mnist fashionmnist jm1 kc1 breastw)
ATTACKS=(loss_threshold shadow_model lira)
MODELS=(xgboost rf)

idx=$SLURM_ARRAY_TASK_ID
ds_idx=$((idx / 6)); rem=$((idx % 6))
DATASET=${DATASETS[$ds_idx]}
ATTACK=${ATTACKS[$((rem / 2))]}
MODEL=${MODELS[$((rem % 2))]}

if [[ -n "$ATTACK_FILTER" && "$ATTACK" != "$ATTACK_FILTER" ]]; then
  echo "skip: attack=$ATTACK filtered out (ATTACK_FILTER=$ATTACK_FILTER)"; exit 0
fi

echo "v2 CPU task $idx -> dataset=$DATASET attack=$ATTACK model=$MODEL seed=$SEED"
python experiments/run_mia_grid.py \
    --dataset "$DATASET" --attack "$ATTACK" --model "$MODEL" \
    --data_dir data/processed --out_dir "$OUT_DIR" \
    --seed "$SEED" --lira_eval_n 2000 --seed_in_name
