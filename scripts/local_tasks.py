"""
Emit the command lines of one pipeline phase for the local (laptop) run of the
#1613 rebuttal experiments, ordered cheap -> expensive so results accumulate
early. Every command is idempotent (each experiment script skips existing
outputs), so a phase can be re-run after an interruption.

    python scripts/local_tasks.py --phase dpri|grid|raw|dp|misc [--data_dir data/processed]
      -> one shell command per line on stdout (consumed by run_local_pipeline.sh)
"""

import argparse
import os
import zipfile
from pathlib import Path

import numpy as np

DATASETS = [
    "adult", "compas", "purchase100", "texas100",
    "nhanes", "movielens", "gowalla",
    "covtype", "digits", "creditg", "spambase", "mushroom", "electricity",
    "letter", "optdigits", "pendigits", "satimage", "segment", "vehicle",
    "ionosphere", "phoneme", "bankmarketing", "magic", "nomao", "har",
    "gasdrift", "mnist", "fashionmnist", "jm1", "kc1", "breastw",
]
ATTACKS = {"loss_threshold": 1.0, "shadow_model": 4.0, "lira": 16.0}
MODELS = {"mlp": 3.0, "xgboost": 1.5, "rf": 2.0}
ENC = ["adult", "compas", "mushroom", "creditg", "bankmarketing", "nomao"]
RECIPE = ["adult", "covtype", "nomao", "bankmarketing", "letter", "electricity", "magic", "mnist", "gowalla"]


def shape_of(npz_path):
    """(n, d) of X inside a compressed .npz without loading the array."""
    with zipfile.ZipFile(npz_path) as zf:
        with zf.open("X.npy") as f:
            version = np.lib.format.read_magic(f)
            shape, _, _ = np.lib.format._read_array_header(f, version)
    return shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["dpri", "grid", "raw", "dp", "misc"])
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--big_threshold", type=float, default=5e7, help="n*d above which a dataset is big (purchase100, texas100)")
    ap.add_argument("--only", choices=["small", "big", "all"], default="all")
    args = ap.parse_args()
    d = Path(args.data_dir)
    sizes = {}
    for name in DATASETS:
        p = d / f"{name}.npz"
        if p.exists():
            n, dim = shape_of(p)
            sizes[name] = (n, dim, n * dim)
    def keep(name):
        big = sizes[name][2] >= args.big_threshold
        return args.only == "all" or (args.only == "big") == big
    names = [n for n in sorted(sizes, key=lambda k: sizes[k][2]) if keep(n)]
    py = "python"
    lines = []
    if args.phase == "dpri":
        for n in names:
            lines.append((sizes[n][2], f"{py} experiments/run_dpri.py --dataset {n} --data_dir {args.data_dir} --out_dir results/dpri"))
    elif args.phase == "grid":
        for n in names:
            for a, wa in ATTACKS.items():
                for m, wm in MODELS.items():
                    w = sizes[n][2] * wa * wm
                    lines.append((w, f"{py} experiments/run_mia_grid.py --dataset {n} --attack {a} --model {m} --data_dir {args.data_dir} "
                                     f"--out_dir results/mia_grid_v2 --seed 42 --lira_eval_n 2000 --seed_in_name"))
    elif args.phase == "raw":
        for n in names:
            lines.append((sizes[n][2], f"{py} experiments/rebuttal_raw_features.py --dataset {n} --data_dir {args.data_dir} --n_jobs 2"))
    elif args.phase == "dp":
        for n in names:
            lines.append((sizes[n][2], f"{py} experiments/rebuttal_dp.py --dataset {n} --data_dir {args.data_dir} --epsilons 1,4,8 --epochs 30 --seed 42"))
    elif args.phase == "misc":
        for n in ENC:
            if n in sizes and keep(n):
                lines.append((sizes[n][2], f"{py} experiments/rebuttal_encoding.py --dataset {n} --raw_dir data/raw"))
        for n in RECIPE:
            if n in sizes and keep(n):
                lines.append((sizes[n][2] * 4, f"{py} experiments/rebuttal_benchmark_recipe.py --dataset {n} --data_dir {args.data_dir}"))
    for _, cmd in sorted(lines, key=lambda t: t[0]):
        print(cmd)


if __name__ == "__main__":
    main()
