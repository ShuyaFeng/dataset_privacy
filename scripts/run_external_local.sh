#!/bin/bash
# Local runner for the pre-registered external corpus (see external_corpus/README.md).
#   nohup bash scripts/run_external_local.sh > logs/external.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
export XGB_NJOBS=3 RF_NJOBS=3
mkdir -p logs/tasks
P=${P:-4}
python experiments/external_corpus.py download
python experiments/external_corpus.py features
run_task() {
  local cmd="$1"; local slug; slug=$(echo "$cmd" | sed -E 's/.*--dataset ([a-z0-9]+) --attack ([a-z_]+) --model ([a-z]+).*/ext_\1_\2_\3/')
  local t0=$SECONDS
  if bash -c "$cmd" > "logs/tasks/$slug.log" 2>&1; then echo "DONE $((SECONDS-t0))s $cmd" >> logs/progress.log
  else echo "FAIL $((SECONDS-t0))s $cmd" >> logs/progress.log; fi
}
export -f run_task
echo "=== external grid started $(date) ==="
python - <<'PY' > logs/external_tasks.txt
import zipfile, numpy as np
from pathlib import Path
names = [l.strip() for l in open("external_corpus/datasets.txt") if l.strip()]
ATT = {"loss_threshold": 1.0, "shadow_model": 4.0, "lira": 16.0}; MOD = {"mlp": 3.0, "xgboost": 1.5, "rf": 2.0}
def shape(p):
    with zipfile.ZipFile(p) as zf:
        with zf.open("X.npy") as f:
            v = np.lib.format.read_magic(f); s, _, _ = np.lib.format._read_array_header(f, v)
    return s
rows = []
for d in names:
    p = Path("data/external") / f"{d}.npz"
    if not p.exists(): continue
    n, dim = shape(p)
    for a, wa in ATT.items():
        for m, wm in MOD.items():
            rows.append((n * dim * wa * wm, f"python experiments/run_mia_grid.py --dataset {d} --attack {a} --model {m} --data_dir data/external --out_dir results/external/mia_grid --seed 42 --lira_eval_n 2000 --seed_in_name"))
for w, c in sorted(rows): print(c)
PY
echo "tasks: $(wc -l < logs/external_tasks.txt)"
xargs -P "$P" -L 1 bash -c 'run_task "$*"' _ < logs/external_tasks.txt
echo "=== external grid complete $(date) ==="
python experiments/external_corpus.py evaluate
