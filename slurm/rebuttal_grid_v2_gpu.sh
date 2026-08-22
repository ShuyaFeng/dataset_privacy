#!/bin/bash
# ── Rebuttal grid v2, GPU part: model=mlp, all attacks, all datasets ────────
# Same configs as the submission (seed 42) but ALSO saves TPR@{10,1,0.1}%FPR,
# train/test accuracy (generalization gap) and uses 2000 LiRA targets.
# 31 datasets x 3 attacks = 93 tasks.
#
# Submit:            sbatch slurm/rebuttal_grid_v2_gpu.sh
# Extra seeds (loss-threshold only, for seed error bars):
#                    sbatch --export=ALL,SEED=43,ATTACK_FILTER=loss_threshold slurm/rebuttal_grid_v2_gpu.sh
#                    sbatch --export=ALL,SEED=44,ATTACK_FILTER=loss_threshold slurm/rebuttal_grid_v2_gpu.sh

#SBATCH --job-name=dpri_v2_gpu
#SBATCH --partition=pascalnodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-92
#SBATCH --output=logs/v2_gpu_%A_%a.out
#SBATCH --error=logs/v2_gpu_%A_%a.err

set -e
mkdir -p logs
cd "$SLURM_SUBMIT_DIR"

module load Anaconda3 2>/dev/null || module load Miniconda3 2>/dev/null || echo "WARNING: no anaconda module"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dpri

SEED=${SEED:-42}
OUT_DIR=${OUT_DIR:-results/mia_grid_v2}
ATTACK_FILTER=${ATTACK_FILTER:-}      # empty = all attacks

DATASETS=(adult compas purchase100 texas100 nhanes movielens gowalla covtype digits creditg spambase mushroom electricity letter optdigits pendigits satimage segment vehicle ionosphere phoneme bankmarketing magic nomao har gasdrift mnist fashionmnist jm1 kc1 breastw)
ATTACKS=(loss_threshold shadow_model lira)
MODEL=mlp

idx=$SLURM_ARRAY_TASK_ID
DATASET=${DATASETS[$((idx / 3))]}
ATTACK=${ATTACKS[$((idx % 3))]}

if [[ -n "$ATTACK_FILTER" && "$ATTACK" != "$ATTACK_FILTER" ]]; then
  echo "skip: attack=$ATTACK filtered out (ATTACK_FILTER=$ATTACK_FILTER)"; exit 0
fi

echo "v2 GPU task $idx -> dataset=$DATASET attack=$ATTACK model=$MODEL seed=$SEED"
python experiments/run_mia_grid.py \
    --dataset "$DATASET" --attack "$ATTACK" --model "$MODEL" \
    --data_dir data/processed --out_dir "$OUT_DIR" \
    --seed "$SEED" --lira_eval_n 2000 --seed_in_name
