#!/bin/bash
# Camera-ready (#1613): per-dataset error bars on Risk(D) from repeated
# member/non-member splits. Re-runs the attack grid with seeds 43 and 44
# (the split, the target models and the shadow models all follow the seed),
# writing results/mia_grid_v2/<dataset>__<attack>__<model>__seed<S>.json.
# The four prohibitive configurations (LiRA and shadow with XGBoost on
# Purchase100 and Texas100; 2-14 h each) are skipped; every other task is
# idempotent (run_mia_grid.py skips existing outputs).
#   nohup bash scripts/run_error_bars.sh > logs/error_bars.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
export XGB_NJOBS=3 RF_NJOBS=3
mkdir -p logs/tasks
P=${P:-4}
run_task() {
  local cmd="$1"; local slug; slug=$(echo "$cmd" | sed -E 's/.*--dataset ([a-z0-9]+) --attack ([a-z_]+) --model ([a-z]+).*--seed ([0-9]+).*/\1_\2_\3_seed\4/')
  local t0=$SECONDS
  if bash -c "$cmd" > "logs/tasks/errbar_$slug.log" 2>&1; then echo "DONE $((SECONDS-t0))s $cmd" >> logs/progress.log
  else echo "FAIL $((SECONDS-t0))s $cmd" >> logs/progress.log; fi
}
export -f run_task
echo "=== error-bar seeds started $(date) ==="
python - <<'PY' > logs/error_bar_tasks.txt
import zipfile, numpy as np
from pathlib import Path
DATASETS = ["adult","compas","purchase100","texas100","nhanes","movielens","gowalla","covtype","digits","creditg","spambase","mushroom","electricity","letter","optdigits","pendigits","satimage","segment","vehicle","ionosphere","phoneme","bankmarketing","magic","nomao","har","gasdrift","mnist","fashionmnist","jm1","kc1","breastw"]
ATT = {"loss_threshold": 1.0, "shadow_model": 4.0, "lira": 16.0}; MOD = {"mlp": 3.0, "xgboost": 1.5, "rf": 2.0}
SKIP = {("purchase100","lira","xgboost"), ("purchase100","shadow_model","xgboost"), ("texas100","lira","xgboost"), ("texas100","shadow_model","xgboost")}
def shape(p):
    with zipfile.ZipFile(p) as zf:
        with zf.open("X.npy") as f:
            v = np.lib.format.read_magic(f); s, _, _ = np.lib.format._read_array_header(f, v)
    return s
rows = []
for d in DATASETS:
    n, dim = shape(Path("data/processed") / f"{d}.npz")
    for a, wa in ATT.items():
        for m, wm in MOD.items():
            if (d, a, m) in SKIP: continue
            for seed in (43, 44):
                rows.append((n * dim * wa * wm, f"python experiments/run_mia_grid.py --dataset {d} --attack {a} --model {m} --data_dir data/processed --out_dir results/mia_grid_v2 --seed {seed} --lira_eval_n 2000 --seed_in_name"))
for w, c in sorted(rows): print(c)
PY
echo "tasks: $(wc -l < logs/error_bar_tasks.txt)"
xargs -P "$P" -L 1 bash -c 'run_task "$*"' _ < logs/error_bar_tasks.txt
echo "=== error-bar seeds complete $(date) ==="
