#!/bin/bash
# Wait until grid (279), DP (31) and DP-LiRA (29) are complete and no producer/aggregator is running,
# then run post (if the ground truth is stale) and the final full aggregation.
cd "$(dirname "$0")/.."; source .venv/bin/activate; export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
while true; do
  g=$(ls results/mia_grid_v2/*.json 2>/dev/null | wc -l | tr -d ' '); d=$(ls results/rebuttal/dp/*.json 2>/dev/null | wc -l | tr -d ' '); l=$(ls results/rebuttal/dp_lira/*.json 2>/dev/null | wc -l | tr -d ' ')
  busy=$(ps -axo command | grep -E "run_mia_grid|rebuttal_dp\.py|rebuttal_experiments\.py|run_regression" | grep -v grep | wc -l | tr -d ' ')
  echo "$(date '+%H:%M') grid=$g/279 dp=$d/31 dp_lira=$l/29 busy=$busy"
  if [ "$g" -ge 279 ] && [ "$d" -ge 31 ] && [ "$l" -ge 29 ] && [ "$busy" -eq 0 ]; then break; fi
  sleep 300
done
newest_grid=$(ls -t results/mia_grid_v2/*.json | head -1)
if [ ! -e results/ground_truth_risk.csv ] || [ "$newest_grid" -nt results/ground_truth_risk.csv ]; then
  echo "ground truth stale -> post"; bash scripts/run_local_pipeline.sh post
fi
echo "=== final aggregate $(date) ==="
python experiments/rebuttal_experiments.py --all --n_boot 10000 --n_perm 500 --n_sub 200 > logs/tasks/aggregate_final.log 2>&1
cp results/rebuttal/rebuttal_summary.md results/rebuttal/rebuttal_summary_final.md
echo "=== FINAL AGGREGATE DONE $(date) ==="; tail -3 logs/tasks/aggregate_final.log
