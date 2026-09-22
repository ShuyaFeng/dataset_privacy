#!/bin/bash
# Side runner: phases that do not depend on the attack grid (raw, misc, dp,
# dp_lira for adult). Every task skips existing outputs, so the main runner
# (scripts/run_labelfix_rerun.sh) will find them done and move on.
set -u
cd "$(dirname "$0")/.."
export P_SMALL=2 P_BIG=1 P_DP=1 XGB_NJOBS=2 RF_NJOBS=2
echo "=== side runner started $(date) ==="
for phase in raw misc dp; do bash scripts/run_local_pipeline.sh "$phase"; done
source .venv/bin/activate
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
if [ ! -f results/rebuttal/dp_lira/adult.json ]; then
  python experiments/rebuttal_dp.py --dataset adult --epsilons 1,8,64 --epochs 100 --lira --lira_shadows 8 \
         --out_dir results/rebuttal/dp_lira --seed 42 > logs/tasks/dp_lira_dataset_adult_labelfix.log 2>&1 \
    && echo "DONE dp_lira adult (side)" || echo "FAIL dp_lira adult (side)"
fi
echo "=== side runner complete $(date) ==="
