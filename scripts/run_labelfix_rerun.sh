#!/bin/bash
# Camera-ready (#1613): re-run everything that depends on the Adult / Texas100
# labels after the loader fix (scripts/download_data.py). Every phase skips
# outputs that already exist, so only the two datasets' tasks actually run.
#   nohup bash scripts/run_labelfix_rerun.sh > logs/labelfix_rerun.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
export P_SMALL=4 P_BIG=2 P_DP=1 XGB_NJOBS=4 RF_NJOBS=4
echo "=== label-fix rerun started $(date) ==="
for phase in dpri grid post raw misc dp; do
  bash scripts/run_local_pipeline.sh "$phase"
done
echo "=== dp_lira (adult; 100 epochs, 8 DP shadows) $(date) ==="
source .venv/bin/activate
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
python experiments/rebuttal_dp.py --dataset adult --epsilons 1,8,64 --epochs 100 --lira --lira_shadows 8 \
       --out_dir results/rebuttal/dp_lira --seed 42 > logs/tasks/dp_lira_dataset_adult_labelfix.log 2>&1 \
  && echo "DONE dp_lira adult" || echo "FAIL dp_lira adult"
bash scripts/run_local_pipeline.sh aggregate
echo "=== camera-ready pending checks $(date) ==="
python experiments/camera_ready_pending.py --stage all --draws 1000 > results/camera_ready/run.log 2>&1 \
  && echo "DONE camera_ready_pending" || echo "FAIL camera_ready_pending"
echo "=== label-fix rerun complete $(date) ==="
