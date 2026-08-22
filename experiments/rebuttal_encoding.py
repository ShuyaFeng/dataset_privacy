"""
Rebuttal (S&P 2027 #1613): categorical-encoding sensitivity.

Reviewer B: "using one-hot encoding instead of integer-encoding for
categorical features might have an impact on memorization and thus on the
success of MIAs."

Fact check on the submitted pipeline (scripts/download_data.py): the four
original-source loaders (Adult, COMPAS, MovieLens, Gowalla) integer-encode
categoricals with LabelEncoder, while every OpenML loader one-hot encodes
them with pd.get_dummies.  So the corpus already mixes both encodings, and
the paper's sentence "categorical features ... are integer-encoded" must be
corrected in the revision.  This script measures how much the encoding
actually moves (a) the DPRI features and (b) the measured risk, on the
datasets that have categorical columns, by rebuilding each under the
OTHER encoding.

  integer -> one-hot : adult, compas
  one-hot -> integer : mushroom, creditg, bankmarketing, nomao

Per dataset, per encoding: DPRI features (k=5) and loss-threshold AUC for
MLP / XGBoost / RF (one target model each, seed 42, same split protocol).

Usage:
    python experiments/rebuttal_encoding.py --dataset adult
    python experiments/rebuttal_encoding.py --dataset mushroom
Output:
    results/rebuttal/encoding/{dataset}.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.dpri.features import compute_dpri_features          # noqa: E402
from src.mia.attacks import attack_loss_threshold, tpr_at_fpr  # noqa: E402
from src.models.classifiers import get_model                 # noqa: E402

OUT_DIR = Path("results/rebuttal/encoding")
OPENML_IDS = {"mushroom": 24, "creditg": 31, "bankmarketing": 1461, "nomao": 1486}
SUBSAMPLE = 30000
MODELS = ["mlp", "xgboost", "rf"]


# ── builders: return {encoding_name: (X, y)} ────────────────────────────────

def _finish(X, y, seed=42):
    X = np.nan_to_num(np.asarray(X, dtype=np.float32))
    _, y = np.unique(np.asarray(y).astype(str), return_inverse=True)
    y = y.astype(np.int32)
    if len(X) > SUBSAMPLE:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), SUBSAMPLE, replace=False)
        X, y = X[idx], y[idx]
    return StandardScaler().fit_transform(X), y


def build_adult(raw_dir):
    cols = ["age", "workclass", "fnlwgt", "education", "education-num",
            "marital-status", "occupation", "relationship", "race", "sex",
            "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]
    tr = pd.read_csv(raw_dir / "adult.data", names=cols, na_values=" ?", skipinitialspace=True)
    te = pd.read_csv(raw_dir / "adult.test", names=cols, na_values=" ?", skipinitialspace=True, skiprows=1)
    df = pd.concat([tr, te], ignore_index=True).dropna()
    df["income"] = df["income"].astype(str).str.replace(".", "", regex=False)
    y = df["income"]
    F = df.drop("income", axis=1)
    cat = list(F.select_dtypes("object").columns)
    F_int = F.copy()
    for c in cat:
        F_int[c] = LabelEncoder().fit_transform(F_int[c].astype(str))
    F_oh = pd.get_dummies(F, columns=cat)
    return {"integer": _finish(F_int.values, y), "onehot": _finish(F_oh.values, y)}


def build_compas(raw_dir):
    df = pd.read_csv(raw_dir / "compas.csv")
    keep = ["age", "c_charge_degree", "race", "sex", "priors_count",
            "days_b_screening_arrest", "decile_score", "two_year_recid"]
    df = df[keep].dropna()
    y = df["two_year_recid"]
    F = df.drop("two_year_recid", axis=1)
    cat = list(F.select_dtypes("object").columns)
    F_int = F.copy()
    for c in cat:
        F_int[c] = LabelEncoder().fit_transform(F_int[c].astype(str))
    F_oh = pd.get_dummies(F, columns=cat)
    return {"integer": _finish(F_int.values, y), "onehot": _finish(F_oh.values, y)}


def build_openml(name):
    from sklearn.datasets import fetch_openml
    d = fetch_openml(data_id=OPENML_IDS[name], as_frame=True, parser="auto")
    F = d.data
    cat = [c for c in F.columns if str(F[c].dtype) in ("category", "object", "bool")]
    F_oh = pd.get_dummies(F, dummy_na=True).apply(pd.to_numeric, errors="coerce").fillna(0)
    F_int = F.copy()
    for c in cat:
        F_int[c] = LabelEncoder().fit_transform(F_int[c].astype(str))
    F_int = F_int.apply(pd.to_numeric, errors="coerce").fillna(0)
    return {"onehot": _finish(F_oh.values, d.target), "integer": _finish(F_int.values, d.target)}


BUILDERS = {"adult": build_adult, "compas": build_compas,
            **{k: (lambda n=k: (lambda raw: build_openml(n)))() for k in OPENML_IDS}}


# ── per-encoding measurement ────────────────────────────────────────────────

def measure(X, y, seed=42):
    res = {"n_samples": int(len(X)), "n_features": int(X.shape[1])}
    res["dpri"] = compute_dpri_features(X, y, k=5)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, random_state=seed, stratify=y)
    n_classes = len(np.unique(y))
    res["mia"] = {}
    for m in MODELS:
        t0 = time.time()
        model = get_model(m, n_classes=n_classes, seed=seed).fit(Xtr, ytr)
        fpr, tpr, auc, det = attack_loss_threshold(model, Xtr, ytr, Xte, yte, return_details=True)
        res["mia"][m] = {"auc": float(auc), "tpr_at_fpr_01": tpr_at_fpr(fpr, tpr, 0.01),
                         "acc_gap": det["train_acc"] - det["test_acc"],
                         "elapsed_sec": round(time.time() - t0, 1)}
        print(f"    {m:<8} AUC={auc:.4f} gap={det['train_acc']-det['test_acc']:+.3f}", flush=True)
    res["mia"]["mean_auc"] = float(np.mean([res["mia"][m]["auc"] for m in MODELS]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(BUILDERS))
    ap.add_argument("--raw_dir", default="data/raw")
    ap.add_argument("--out_dir", default=str(OUT_DIR))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.dataset}.json"
    if out_path.exists():
        print(f"Already done: {out_path}")
        return
    variants = BUILDERS[args.dataset](Path(args.raw_dir))
    out = {"dataset": args.dataset, "seed": args.seed,
           "submitted_encoding": "integer" if args.dataset in ("adult", "compas") else "onehot",
           "encodings": {}}
    for enc, (X, y) in variants.items():
        print(f"  [{args.dataset}] encoding={enc}: X={X.shape}", flush=True)
        out["encodings"][enc] = measure(X, y, args.seed)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
