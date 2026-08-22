"""
End-to-end smoke test of the rebuttal pipeline on SYNTHETIC data.

Builds, in a temporary directory, every input the rebuttal scripts expect
(31 small fake datasets, DPRI features computed with the real code, a fake
MIA grid whose AUC is a noisy function of the geometry, fake v2/dp/encoding/
recipe results in the exact formats the cluster scripts write), then runs
rebuttal_raw_features.py --all and rebuttal_experiments.py --experiment all
with tiny bootstrap counts.  Catches crashes and format mismatches BEFORE
anything is submitted to the cluster.  Takes ~1-2 minutes.

Run:
    python tests/test_rebuttal.py
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.dpri.features import compute_dpri_features  # noqa: E402

DATASETS = [
    "adult", "compas", "purchase100", "texas100",
    "nhanes", "movielens", "gowalla",
    "covtype", "digits", "creditg", "spambase", "mushroom", "electricity",
    "letter", "optdigits", "pendigits", "satimage", "segment", "vehicle",
    "ionosphere", "phoneme", "bankmarketing", "magic", "nomao", "har",
    "gasdrift", "mnist", "fashionmnist", "jm1", "kc1", "breastw",
]
ATTACKS = ["loss_threshold", "shadow_model", "lira"]
MODELS = ["mlp", "xgboost", "rf"]


def make_fixture(td: Path):
    rng = np.random.default_rng(0)
    (td / "data/processed").mkdir(parents=True)
    (td / "results/dpri").mkdir(parents=True)
    (td / "results/mia_grid").mkdir(parents=True)
    (td / "results/mia_grid_v2").mkdir(parents=True)
    (td / "results/rebuttal/dp").mkdir(parents=True)
    (td / "results/rebuttal/encoding").mkdir(parents=True)
    (td / "results/rebuttal/recipe").mkdir(parents=True)
    feats, risk_rows = [], []
    for i, name in enumerate(DATASETS):
        n = int(rng.integers(300, 1500))
        d = int(rng.choice([5, 8, 12, 20, 40, 80, 120]))
        c = int(min(rng.choice([2, 2, 2, 3, 5, 10]), d))
        X = rng.normal(size=(n, d)).astype(np.float32) * rng.uniform(0.3, 3.0)
        if name in ("gowalla", "compas"):            # many exact duplicates
            X[: n // 2] = X[0]
        y = (np.argmax(X[:, :c] + rng.normal(0, 1.5, size=(n, c)), axis=1)).astype(np.int32)
        if name == "mushroom":                       # separable task
            y = (X[:, 0] > 0).astype(np.int32)
        np.savez_compressed(td / "data/processed" / f"{name}.npz", X=X, y=y)
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            f = compute_dpri_features(X, y, k=5)
        f["dataset"] = name
        f["n_samples"] = n
        f["n_features"] = d
        feats.append(f)
        # fake ground truth: noisy function of geometry so correlations exist
        base = 0.55 + 0.25 * np.tanh(0.3 * np.log1p(f["uniqueness_mean"]) - 0.1 * f["cluster_sep"]) + rng.normal(0, 0.04)
        for a in ATTACKS:
            for m in MODELS:
                auc = float(np.clip(base + {"lira": 0.08, "loss_threshold": 0, "shadow_model": -0.02}[a]
                                    + {"rf": 0.15, "xgboost": 0.02, "mlp": -0.05}[m] + rng.normal(0, 0.02), 0.45, 1.0))
                rec = {"dataset": name, "attack": a, "model": m, "auc": auc, "n_samples": n,
                       "n_features": d, "n_classes": c, "seed": 42, "elapsed_sec": 1.0}
                json.dump(rec, open(td / "results/mia_grid" / f"{name}__{a}__{m}.json", "w"))
                for seed in (42, 43, 44):
                    if seed != 42 and a != "loss_threshold":
                        continue
                    r2 = dict(rec, seed=seed, auc=float(np.clip(auc + rng.normal(0, 0.01), 0.45, 1)),
                              tpr_at_fpr_10=float(np.clip(auc - 0.3 + rng.normal(0, 0.03), 0, 1)),
                              tpr_at_fpr_01=float(np.clip(auc - 0.45 + rng.normal(0, 0.03), 0, 1)),
                              tpr_at_fpr_001=float(np.clip(auc - 0.5 + rng.normal(0, 0.02), 0, 1)),
                              train_acc=0.9, test_acc=float(0.9 - (auc - 0.5)), acc_gap=float(auc - 0.5), n_scores=1000)
                    json.dump(r2, open(td / "results/mia_grid_v2" / f"{name}__{a}__{m}__s{seed}.json", "w"))
        # fake DP results (only some datasets, to exercise the partial path)
        if i % 2 == 0:
            res = {"baseline": {"auc": base, "test_acc": 0.8, "tpr_at_fpr_01": 0.1}}
            for e in (1.0, 4.0, 16.0):
                res[f"eps_{e:g}"] = {"auc": float(0.5 + (base - 0.5) * (e / 20)), "test_acc": float(0.8 - 0.2 / e),
                                     "tpr_at_fpr_01": 0.02, "epsilon_target": e, "epsilon_spent": e}
            json.dump({"dataset": name, "results": res}, open(td / "results/rebuttal/dp" / f"{name}_dp.json", "w"))
    pd.DataFrame(feats).set_index("dataset").to_csv(td / "results/dpri/dpri_features.csv")
    grid = pd.DataFrame([json.load(open(p)) for p in (td / "results/mia_grid").glob("*.json")])
    grid.groupby("dataset")["auc"].mean().rename("Risk_D").reset_index().to_csv(td / "results/ground_truth_risk.csv", index=False)
    # fake encoding + recipe
    for name in ("adult", "mushroom"):
        encs = {}
        for enc in ("integer", "onehot"):
            encs[enc] = {"n_samples": 1000, "n_features": 14 if enc == "integer" else 100,
                         "dpri": {"uniqueness_mean": 1.0 + rng.random(), "density_mean": rng.random(), "cluster_sep": 0.01,
                                  "outlier_mean": 0.3, "entropy": 3.0},
                         "mia": {m: {"auc": 0.6 + 0.1 * rng.random(), "tpr_at_fpr_01": 0.05, "acc_gap": 0.1} for m in MODELS}}
            encs[enc]["mia"]["mean_auc"] = float(np.mean([encs[enc]["mia"][m]["auc"] for m in MODELS]))
        json.dump({"dataset": name, "submitted_encoding": "integer" if name == "adult" else "onehot", "encodings": encs},
                  open(td / "results/rebuttal/encoding" / f"{name}.json", "w"))
    for name in ("adult", "covtype"):
        v = {}
        for k, uscale in [("original", 1.0), ("step1_cluster", 1.0), ("step2_cluster_balance", 2.5), ("subsample_only_control", 1.3)]:
            v[k] = {"geometry": {"uniqueness_mean": uscale, "density_mean": 1 / uscale, "cluster_sep": 0.05 * uscale,
                                 "n_samples": 20000, "n_classes": 100 if "cluster" in k else 2},
                    "risk": {"mean_auc": 0.6 + 0.05 * uscale, "mean_auc_regularized": 0.55 + 0.05 * uscale}}
        json.dump({"dataset": name, "variants": v}, open(td / "results/rebuttal/recipe" / f"{name}.json", "w"))


def run(cmd, cwd):
    print("  $", " ".join(cmd))
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
        raise SystemExit(f"FAILED: {' '.join(cmd)}")
    return r.stdout


def main():
    with tempfile.TemporaryDirectory() as tds:
        td = Path(tds)
        print("building fixture ...")
        make_fixture(td)
        env = dict(os.environ, PYTHONPATH=str(ROOT))
        py = sys.executable
        print("raw features on all fixture datasets ...")
        out = subprocess.run([py, str(ROOT / "experiments/rebuttal_raw_features.py"), "--all", "--n_jobs", "2"],
                             cwd=td, capture_output=True, text=True, env=env)
        if out.returncode != 0:
            print(out.stdout[-2000:], out.stderr[-3000:])
            raise SystemExit("FAILED: rebuttal_raw_features.py")
        n_raw = len(list((td / "results/rebuttal/raw").glob("*_raw.json")))
        assert n_raw == len(DATASETS), f"raw outputs: {n_raw}"
        print(f"  {n_raw} raw json written")
        print("aggregate analyses (tiny bootstrap) ...")
        out = subprocess.run([py, str(ROOT / "experiments/rebuttal_experiments.py"), "--experiment", "all",
                              "--n_boot", "20", "--n_perm", "10", "--n_jobs", "2"],
                             cwd=td, capture_output=True, text=True, env=env)
        print(out.stdout[-6000:])
        if out.returncode != 0:
            print(out.stderr[-4000:])
            raise SystemExit("FAILED: rebuttal_experiments.py")
        md = td / "results/rebuttal/analysis/REBUTTAL_NUMBERS.md"
        assert md.exists()
        txt = md.read_text()
        missing = [s for s in ["nested CV", "error bars", "theorem formula", "ground-truth robustness",
                               "eta^2", "triage", "failure analysis", "sensitivity", "in-sample vs out-of-sample",
                               "Gowalla vs Movielens", "label disagreement", "TPR at low FPR", "DP-trained",
                               "encoding sensitivity", "benchmark construction recipe"] if s not in txt]
        assert not missing, f"sections missing from report: {missing}"
        figs = sorted(p.name for p in (td / "results/rebuttal/analysis/figures").glob("*.png"))
        print("  figures:", figs)
        assert "bootstrap_ci.png" in figs and "u_in_vs_u_out.png" in figs
        # the headline re-implementation must match the paper code (asserted inside exp_nested)
        nested = json.load(open(td / "results/rebuttal/analysis/nested.json"))
        assert abs(nested["nested_spearman"] - nested["paper_code_nested"]) < 1e-9
        print(f"\nPASS test_rebuttal (nested rho on fixture = {nested['nested_spearman']:.3f})")


if __name__ == "__main__":
    main()
