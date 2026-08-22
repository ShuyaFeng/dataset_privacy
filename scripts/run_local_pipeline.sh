#!/bin/bash
# Local (laptop) end-to-end run of the #1613 rebuttal experiments.
#   bash scripts/run_local_pipeline.sh            # everything, idempotent / resumable
#   bash scripts/run_local_pipeline.sh grid       # one phase: dpri | grid | post | raw | misc | dp | aggregate
# Logs: logs/pipeline.log (phase markers), logs/progress.log (one DONE/FAIL line per task),
#       logs/tasks/<slug>.log (per-task stdout+stderr).
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
mkdir -p logs/tasks results/mia_grid_v2 results/dpri results/regression results/rebuttal
P_SMALL=${P_SMALL:-5}; P_BIG=${P_BIG:-3}; P_DP=${P_DP:-2}

run_task() {   # $1 = command
  local cmd="$1"; local slug; slug=$(echo "$cmd" | sed -E 's/.*experiments\/([a-z_]+)\.py//; s/[^a-zA-Z0-9]+/_/g; s/^_//; s/_$//')
  slug="$(echo "$cmd" | grep -oE 'experiments/[a-z_]+' | sed 's#experiments/##')_$slug"
  local t0=$SECONDS
  if bash -c "$cmd" > "logs/tasks/$slug.log" 2>&1; then
    echo "DONE $((SECONDS-t0))s $cmd" >> logs/progress.log
  else
    echo "FAIL $((SECONDS-t0))s $cmd" >> logs/progress.log
  fi
}
export -f run_task

run_queue() {  # $1 = phase, $2 = only(small|big|all), $3 = parallelism
  local n; n=$(python scripts/local_tasks.py --phase "$1" --only "$2" | wc -l | tr -d ' ')
  echo "=== phase $1 ($2): $n tasks, P=$3, $(date) ===" | tee -a logs/pipeline.log
  # -L 1 (not -I) avoids BSD xargs' 255-byte replacement limit; "$*" re-joins the line
  python scripts/local_tasks.py --phase "$1" --only "$2" | xargs -P "$3" -L 1 bash -c 'run_task "$*"' _
  echo "=== phase $1 ($2) finished $(date) ===" | tee -a logs/pipeline.log
}

phase=${1:-all}
if [[ $phase == all || $phase == dpri ]]; then run_queue dpri small $P_SMALL; run_queue dpri big $P_BIG; fi
if [[ $phase == all || $phase == grid ]]; then run_queue grid small $P_SMALL; run_queue grid big $P_BIG; fi
if [[ $phase == all || $phase == post ]]; then
  echo "=== post: merge DPRI, ground truth, regression $(date) ===" | tee -a logs/pipeline.log
  python experiments/run_dpri.py --merge > logs/tasks/post_merge.log 2>&1
  # check_ground_truth_variance.py reads results/mia_grid; point it at the v2 grid via a symlink if the old dir is empty
  if [ ! -e results/mia_grid ] || [ -z "$(ls -A results/mia_grid 2>/dev/null)" ]; then rm -rf results/mia_grid; ln -s mia_grid_v2 results/mia_grid; fi
  python experiments/check_ground_truth_variance.py > logs/tasks/post_ground_truth.log 2>&1
  python experiments/run_regression.py > logs/tasks/post_regression.log 2>&1
  tail -12 logs/tasks/post_regression.log | tee -a logs/pipeline.log
fi
if [[ $phase == all || $phase == raw ]]; then run_queue raw small $P_SMALL; run_queue raw big $P_BIG; fi
if [[ $phase == all || $phase == misc ]]; then run_queue misc all $P_BIG; fi
if [[ $phase == all || $phase == dp ]]; then run_queue dp small $P_DP; run_queue dp big 1; fi
if [[ $phase == all || $phase == aggregate ]]; then
  echo "=== aggregate $(date) ===" | tee -a logs/pipeline.log
  python experiments/rebuttal_experiments.py --all --n_boot 10000 --n_perm 1000 --n_sub 200 > logs/tasks/aggregate.log 2>&1
  tail -5 logs/tasks/aggregate.log | tee -a logs/pipeline.log
fi
echo "=== pipeline ($phase) complete $(date) ===" | tee -a logs/pipeline.log
