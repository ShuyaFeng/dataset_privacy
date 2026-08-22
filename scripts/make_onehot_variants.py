"""
Rebuttal (#1613), Reviewer B (encoding): one-hot variants of the two datasets
whose categorical columns are integer-encoded in the main pipeline (Adult, COMPAS).
All OpenML datasets already go through pd.get_dummies in download_data.py.

    python scripts/make_onehot_variants.py [--raw_dir data/raw] [--out_dir data/processed]
    -> data/processed/adult_onehot.npz, data/processed/compas_onehot.npz

Then:  python experiments/run_dpri.py --dataset adult_onehot
       python experiments/run_mia_grid.py --dataset adult_onehot --attack ... --model ...
"""

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("download_data", HERE / "download_data.py")
dl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dl)


def _finish(df: pd.DataFrame, label: str, out_path: Path):
    y = LabelEncoder().fit_transform(df[label].astype(str)).astype(np.int32)
    Xdf = df.drop(columns=[label])
    cat_cols = list(Xdf.select_dtypes("object").columns)
    Xdf = pd.get_dummies(Xdf, columns=cat_cols, dummy_na=False)
    X = Xdf.apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)
    X = StandardScaler().fit_transform(X)
    dl._save(out_path, X, y)
    print(f"  saved {out_path}.npz  shape={X.shape}  (one-hot columns from {len(cat_cols)} categoricals)")


def adult_onehot(raw_dir: Path, out_dir: Path):
    print("Adult (one-hot)")
    dl.download("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", raw_dir / "adult.data")
    dl.download("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test", raw_dir / "adult.test")
    cols = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
            "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
            "hours-per-week", "native-country", "income"]
    train = pd.read_csv(raw_dir / "adult.data", names=cols, na_values=" ?", skipinitialspace=True)
    test = pd.read_csv(raw_dir / "adult.test", names=cols, na_values=" ?", skipinitialspace=True, skiprows=1)
    df = pd.concat([train, test], ignore_index=True).dropna()
    df["income"] = df["income"].astype(str).str.replace(".", "", regex=False)
    _finish(df, "income", out_dir / "adult_onehot")


def compas_onehot(raw_dir: Path, out_dir: Path):
    print("COMPAS (one-hot)")
    dest = raw_dir / "compas.csv"
    dl.download("https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv", dest)
    df = pd.read_csv(dest)
    keep = ["age", "c_charge_degree", "race", "sex", "priors_count",
            "days_b_screening_arrest", "decile_score", "two_year_recid"]
    df = df[keep].dropna()
    df["two_year_recid"] = df["two_year_recid"].astype(int)
    _finish(df, "two_year_recid", out_dir / "compas_onehot")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="data/raw")
    ap.add_argument("--out_dir", default="data/processed")
    args = ap.parse_args()
    raw, out = Path(args.raw_dir), Path(args.out_dir)
    raw.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    adult_onehot(raw, out)
    compas_onehot(raw, out)


if __name__ == "__main__":
    main()
