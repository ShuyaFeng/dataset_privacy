"""
Download and preprocess all datasets for the DPRI experiment.
Run once on the cluster after environment setup.

Usage:
    python scripts/download_data.py --data_dir data/raw
"""

import argparse
import os
import zipfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def progress_hook(count, block_size, total_size):
    pct = count * block_size * 100 // total_size
    print(f"\r  {pct}%", end="", flush=True)


def download(url: str, dest: Path):
    if dest.exists():
        print(f"  already exists: {dest.name}")
        return
    print(f"  downloading {dest.name} ...")
    urllib.request.urlretrieve(url, dest, reporthook=progress_hook)
    print()


# ---------------------------------------------------------------------------
# 1. Adult (UCI)
# ---------------------------------------------------------------------------
def load_adult(raw_dir: Path, out_dir: Path):
    print("[1/7] Adult")
    download(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        raw_dir / "adult.data",
    )
    download(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
        raw_dir / "adult.test",
    )
    cols = [
        "age","workclass","fnlwgt","education","education-num",
        "marital-status","occupation","relationship","race","sex",
        "capital-gain","capital-loss","hours-per-week","native-country","income",
    ]
    train = pd.read_csv(raw_dir / "adult.data", names=cols, na_values=" ?", skipinitialspace=True)
    test  = pd.read_csv(raw_dir / "adult.test",  names=cols, na_values=" ?", skipinitialspace=True, skiprows=1)
    df = pd.concat([train, test], ignore_index=True).dropna()
    for c in df.select_dtypes("object").columns:
        df[c] = LabelEncoder().fit_transform(df[c].astype(str))
    X = df.drop("income", axis=1).values.astype(np.float32)
    y = df["income"].values.astype(np.int32)
    X = StandardScaler().fit_transform(X)
    _save(out_dir / "adult", X, y)
    print(f"  saved: {X.shape}")


# ---------------------------------------------------------------------------
# 2. COMPAS (ProPublica)
# ---------------------------------------------------------------------------
def load_compas(raw_dir: Path, out_dir: Path):
    print("[2/7] COMPAS")
    url = (
        "https://raw.githubusercontent.com/propublica/compas-analysis/"
        "master/compas-scores-two-years.csv"
    )
    dest = raw_dir / "compas.csv"
    download(url, dest)
    df = pd.read_csv(dest)
    keep = ["age","c_charge_degree","race","sex","priors_count",
            "days_b_screening_arrest","decile_score","two_year_recid"]
    df = df[keep].dropna()
    for c in df.select_dtypes("object").columns:
        df[c] = LabelEncoder().fit_transform(df[c].astype(str))
    X = df.drop("two_year_recid", axis=1).values.astype(np.float32)
    y = df["two_year_recid"].values.astype(np.int32)
    X = StandardScaler().fit_transform(X)
    _save(out_dir / "compas", X, y)
    print(f"  saved: {X.shape}")


# ---------------------------------------------------------------------------
# 3. Purchase100 (Shokri et al. benchmark)
# ---------------------------------------------------------------------------
def load_purchase100(raw_dir: Path, out_dir: Path):
    print("[3/7] Purchase100")
    dest = raw_dir / "purchase100.npz"
    if not dest.exists():
        print("  NOTE: Purchase100 must be downloaded manually.")
        print("  Visit: https://github.com/privacytrustlab/datasets")
        print("  Download 'dataset_purchase.tgz', extract purchase100.npz, place in data/raw/")
        print("  Skipping for now.")
        return
    data = np.load(dest)
    X = data["features"].astype(np.float32)
    y = data["labels"].astype(np.int32)
    _save(out_dir / "purchase100", X, y)
    print(f"  saved: {X.shape}")


# ---------------------------------------------------------------------------
# 4. Texas100 (Shokri et al. benchmark)
# ---------------------------------------------------------------------------
def load_texas100(raw_dir: Path, out_dir: Path):
    print("[4/7] Texas100")
    dest = raw_dir / "texas100.npz"
    if not dest.exists():
        print("  NOTE: Texas100 must be downloaded manually.")
        print("  Visit: https://github.com/privacytrustlab/datasets")
        print("  Download 'dataset_texas.tgz', extract texas100.npz, place in data/raw/")
        print("  Skipping for now.")
        return
    data = np.load(dest)
    X = data["features"].astype(np.float32)
    y = data["labels"].astype(np.int32)
    _save(out_dir / "texas100", X, y)
    print(f"  saved: {X.shape}")


# ---------------------------------------------------------------------------
# 5. Diabetes (Pima Indians) — medical domain substitute
#    NHANES XPT format is no longer reliably accessible via direct URL.
#    Pima Indians Diabetes is a well-established public medical dataset.
# ---------------------------------------------------------------------------
def load_nhanes(raw_dir: Path, out_dir: Path):
    print("[5/7] Diabetes (Pima Indians) — medical domain")
    dest = raw_dir / "pima_diabetes.csv"
    download(
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
        "pima-indians-diabetes.data.csv",
        dest,
    )
    cols = [
        "pregnancies", "glucose", "blood_pressure", "skin_thickness",
        "insulin", "bmi", "diabetes_pedigree", "age", "outcome",
    ]
    df = pd.read_csv(dest, names=cols).dropna()
    X = df.drop("outcome", axis=1).values.astype(np.float32)
    y = df["outcome"].values.astype(np.int32)
    X = StandardScaler().fit_transform(X)
    _save(out_dir / "nhanes", X, y)   # keep filename for compatibility
    print(f"  saved: {X.shape}")


# ---------------------------------------------------------------------------
# 6. MovieLens-1M
# ---------------------------------------------------------------------------
def load_movielens(raw_dir: Path, out_dir: Path):
    print("[6/7] MovieLens-1M")
    dest_zip = raw_dir / "ml-1m.zip"
    download("https://files.grouplens.org/datasets/movielens/ml-1m.zip", dest_zip)
    dest_dir = raw_dir / "ml-1m"
    if not dest_dir.exists():
        with zipfile.ZipFile(dest_zip) as z:
            z.extractall(raw_dir)
    ratings = pd.read_csv(
        dest_dir / "ratings.dat", sep="::", engine="python",
        names=["user","movie","rating","timestamp"]
    )
    # Treat as tabular: user-level features = rating statistics
    feats = ratings.groupby("user")["rating"].agg(
        ["mean","std","count","min","max"]
    ).fillna(0).reset_index()
    # label: high-activity user (top 50%)
    feats["label"] = (feats["count"] > feats["count"].median()).astype(int)
    X = feats[["mean","std","count","min","max"]].values.astype(np.float32)
    y = feats["label"].values.astype(np.int32)
    X = StandardScaler().fit_transform(X)
    _save(out_dir / "movielens", X, y)
    print(f"  saved: {X.shape}")


# ---------------------------------------------------------------------------
# 7. Gowalla (Stanford SNAP mobility dataset)
# ---------------------------------------------------------------------------
def load_gowalla(raw_dir: Path, out_dir: Path):
    print("[7/7] Gowalla")
    dest = raw_dir / "loc-gowalla_totalCheckins.txt.gz"
    download(
        "https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz",
        dest,
    )
    df = pd.read_csv(dest, sep="\t", names=["user","time","lat","lon","loc"])
    # User-level features: location diversity, check-in frequency
    feats = df.groupby("user").agg(
        lat_std=("lat","std"),
        lon_std=("lon","std"),
        n_checkins=("loc","count"),
        n_unique_locs=("loc","nunique"),
        lat_mean=("lat","mean"),
        lon_mean=("lon","mean"),
    ).fillna(0).reset_index()
    # label: high-mobility user (unique locs > median)
    feats["label"] = (feats["n_unique_locs"] > feats["n_unique_locs"].median()).astype(int)
    X = feats[["lat_std","lon_std","n_checkins","n_unique_locs","lat_mean","lon_mean"]].values.astype(np.float32)
    y = feats["label"].values.astype(np.int32)
    X = StandardScaler().fit_transform(X)
    _save(out_dir / "gowalla", X, y)
    print(f"  saved: {X.shape}")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def _save(path: Path, X: np.ndarray, y: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path) + ".npz", X=X, y=y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/raw")
    parser.add_argument("--out_dir",  default="data/processed")
    args = parser.parse_args()

    raw_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    load_adult(raw_dir, out_dir)
    load_compas(raw_dir, out_dir)
    load_purchase100(raw_dir, out_dir)
    load_texas100(raw_dir, out_dir)
    load_nhanes(raw_dir, out_dir)
    load_movielens(raw_dir, out_dir)
    load_gowalla(raw_dir, out_dir)

    print("\nDone. Check data/processed/ for .npz files.")
    print("Manually download Purchase100 and Texas100 from:")
    print("  https://github.com/privacytrustlab/datasets")


if __name__ == "__main__":
    main()
