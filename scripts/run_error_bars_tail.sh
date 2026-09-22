#!/bin/bash
# Side runner: starts the two most expensive error-bar tasks (Texas100 LiRA with RF,
# ~4.7 h each at seed 42) ahead of their place at the end of the xargs queue in
# run_error_bars.sh, sequentially and at lower priority. run_mia_grid.py skips an
# existing output, so the main queue passes over them once they are done.
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
export XGB_NJOBS=3 RF_NJOBS=3
echo "=== error-bar tail runner started $(date) ==="
for seed in 43 44; do
  cmd="python experiments/run_mia_grid.py --dataset texas100 --attack lira --model rf --data_dir data/processed --out_dir results/mia_grid_v2 --seed $seed --lira_eval_n 2000 --seed_in_name"
  t0=$SECONDS
  if nice -n 5 bash -c "$cmd" > "logs/tasks/errbar_texas100_lira_rf_seed${seed}.log" 2>&1; then echo "DONE $((SECONDS-t0))s $cmd" >> logs/progress.log
  else echo "FAIL $((SECONDS-t0))s $cmd" >> logs/progress.log; fi
done
echo "=== error-bar tail runner complete $(date) ==="
