#!/bin/bash
cd "$(dirname "$0")/.."; source .venv/bin/activate
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
python scripts/local_tasks.py --phase raw --only all | sed 's/$/ --force/' | xargs -P 3 -L 1 bash -c 'echo "RUN $*"; $* && echo "OK $*" || echo "FAIL $*"' _
echo "=== raw rerun complete; aggregating ==="
python experiments/rebuttal_experiments.py --all --n_boot 10000 --n_perm 2000 --n_sub 200 > logs/tasks/aggregate_v2.log 2>&1
cp results/rebuttal/rebuttal_summary.md results/rebuttal/rebuttal_summary_final.md
echo "AGGREGATE V2 DONE"; tail -3 logs/tasks/aggregate_v2.log
