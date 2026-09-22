"""
Camera-ready (#1613, S&P 2027): the pending model-free checks.

Everything here is computed from data/processed/*.npz (the standardized
matrices the DPRI features and the attack grid used), the re-run ground
truth results/ground_truth_risk.csv, results/dpri/dpri_features.csv and the
rebuttal's per-dataset geometry results/rebuttal/raw/*_raw.json. No model
is trained. Stages (each writes its own JSON and is skipped when present):

  w1          Wasserstein-1 between the member and non-member halves of the
              attack split (seed 42, stratified 50/50) vs the nearest-
              neighbour spacing u-bar: the direct check of (A3').
  twofactor   Delta-bar = mean_i delta_i * iota_i with a Gaussian kernel at a
              per-dataset bandwidth (Proposition 2's own statistic).
  classes     number of classes C, chance level 1 - sum_c p_c^2, and the
              chance-normalized k-NN label disagreement.
  complexity  classical data-complexity measures (Ho & Basu; Lorena et al.):
              N1, N2, N3, F1, class imbalance.
  encoding    scale-free uniqueness u-tilde under integer vs one-hot encoding
              on the six re-encoded datasets, and the headline with those
              six datasets' features swapped to the alternative encoding.
  corpus      corpus-size curve of the headline with a fixed seed and more
              draws than the rebuttal's 200.
  summary     Spearman of every new statistic with Risk(D) and the nested-CV
              runs; writes results/camera_ready/SUMMARY.md.

Usage:
    .venv/bin/python experiments/camera_ready_pending.py --stage all
    .venv/bin/python experiments/camera_ready_pending.py --stage w1,twofactor
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from experiments.run_regression import nested_cv_spearman, _rank_ridge_predict  # noqa: E402

DATA = ROOT / "data" / "processed"
RES = ROOT / "results"
OUT = RES / "camera_ready"
OUT.mkdir(parents=True, exist_ok=True)

DATASETS = [
    "adult", "compas", "purchase100", "texas100", "nhanes", "movielens", "gowalla",
    "covtype", "digits", "creditg", "spambase", "mushroom", "electricity", "letter",
    "optdigits", "pendigits", "satimage", "segment", "vehicle", "ionosphere", "phoneme",
    "bankmarketing", "magic", "nomao", "har", "gasdrift", "mnist", "fashionmnist",
    "jm1", "kc1", "breastw",
]
SEED = 42
SUB_KERNEL = 30000      # rows used for the kernel / k-NN statistics
SUB_W1 = 3000           # points per side for the assignment-based W1
SUB_CPLX = 5000         # rows for the complexity measures
FEATURE_COLS = ["uniqueness_mean", "density_mean", "outlier_mean", "entropy", "cluster_sep"]
GEO = ["uniqueness_mean", "density_mean", "cluster_sep"]


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def load_xy(name):
    d = np.load(DATA / f"{name}.npz")
    X = np.asarray(d["X"], dtype=np.float32)
    y = np.asarray(d["y"]).astype(int)
    return X, y


def subsample(X, y, m, seed=SEED):
    if len(X) <= m:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), m, replace=False)
    return X[idx], y[idx]


def load_risk():
    r = pd.read_csv(RES / "ground_truth_risk.csv").set_index("dataset")["Risk_D"]
    return r


def load_features():
    df = pd.read_csv(RES / "dpri" / "dpri_features.csv", index_col=0)
    df["log_nfeatures"] = np.log(df["n_features"].clip(lower=1))
    return df


def load_raw(name):
    p = RES / "rebuttal" / "raw" / f"{name}_raw.json"
    return json.load(open(p)) if p.exists() else {}


def stage_done(stage):
    return (OUT / f"{stage}.json").exists()


def save(stage, obj):
    with open(OUT / f"{stage}.json", "w") as f:
        json.dump(obj, f, indent=1)
    log(f"saved {OUT / (stage + '.json')}")


# ───────────────────────────────────────── W1 vs u-bar ─────────────────────

def w1_assignment(A, B):
    """W1 between two equal-size point clouds = mean cost of the optimal matching."""
    C = cdist(A, B)
    r, c = linear_sum_assignment(C)
    return float(C[r, c].mean())


def stage_w1():
    out = {}
    for name in DATASETS:
        t0 = time.time()
        X, y = load_xy(name)
        try:
            Xm, Xn, ym, yn = train_test_split(X, y, test_size=0.5, random_state=SEED, stratify=y)
        except ValueError:
            Xm, Xn, ym, yn = train_test_split(X, y, test_size=0.5, random_state=SEED)
        m = min(SUB_W1, len(Xm), len(Xn))
        w1s, ubars_sub, uout_sub = [], [], []
        for rep in range(3):
            rng = np.random.default_rng(SEED + rep)
            ia = rng.choice(len(Xm), m, replace=False)
            ib = rng.choice(len(Xn), m, replace=False)
            A, B = Xm[ia].astype(np.float64), Xn[ib].astype(np.float64)
            w1s.append(w1_assignment(A, B))
            # in-sample 5-NN spacing on the same member subsample (the DPRI statistic)
            nn = NearestNeighbors(n_neighbors=6).fit(A)
            dA, _ = nn.kneighbors(A)
            ubars_sub.append(float(dA[:, 5].mean()))
            # nearest non-member (k=1) from the member subsample to the non-member subsample
            nn2 = NearestNeighbors(n_neighbors=1).fit(B)
            dB, _ = nn2.kneighbors(A)
            uout_sub.append(float(dB[:, 0].mean()))
        out[name] = {"m": int(m), "n_members": int(len(Xm)), "d": int(X.shape[1]),
                     "w1_mean": float(np.mean(w1s)), "w1_reps": w1s,
                     "ubar5_subsample": float(np.mean(ubars_sub)),
                     "uout1_subsample": float(np.mean(uout_sub)),
                     "elapsed_sec": round(time.time() - t0, 1)}
        log(f"w1 {name}: W1={out[name]['w1_mean']:.4f} ubar5(sub)={out[name]['ubar5_subsample']:.4f} "
            f"uout1(sub)={out[name]['uout1_subsample']:.4f} ({out[name]['elapsed_sec']}s)")
    save("w1", out)


# ───────────────────────────────────────── two-factor Delta-bar ────────────

def stage_twofactor(k=50):
    out = {}
    for name in DATASETS:
        t0 = time.time()
        X, y = load_xy(name)
        Xs, ys = subsample(X, y, SUB_KERNEL)
        nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", n_jobs=-1).fit(Xs)
        dist, idx = nn.kneighbors(Xs)
        dist, idx = dist[:, 1:], idx[:, 1:]                     # drop self
        r5 = dist[:, 4]
        same = (ys[idx] == ys[:, None])
        # median pairwise distance on a 2000-row subsample (scale-free bandwidth variant)
        Xp, _ = subsample(Xs, ys, 2000, seed=SEED + 1)
        D = cdist(Xp, Xp)
        med_pd = float(np.median(D[np.triu_indices(len(Xp), 1)]))
        res = {"n_used": int(len(Xs)), "d": int(X.shape[1]), "k": k,
               "median_r5": float(np.median(r5)), "median_pairwise": med_pd}
        for tag, h in (("h_median_r5", float(np.median(r5))), ("h_median_pairwise", med_pd),
                       ("h_half_median_pairwise", 0.5 * med_pd)):
            if h <= 0:
                continue
            K = np.exp(-(dist ** 2) / (2 * h * h))
            W = K.sum(1)
            with np.errstate(invalid="ignore", divide="ignore"):
                purity = (K * same).sum(1) / np.maximum(W, 1e-300)
            delta = 1.0 - purity
            iota = 1.0 / (1.0 + W)
            Delta = delta * iota
            res[tag] = {"h": h,
                        "Delta_bar": float(Delta.mean()),
                        "delta_bar_kernel": float(delta.mean()),
                        "iota_bar": float(iota.mean()),
                        "W_mean": float(W.mean()), "W_median": float(np.median(W)),
                        "kernel_weight_at_kth_neighbour_mean": float(K[:, -1].mean()),
                        "frac_rows_W_below_1e-3": float((W < 1e-3).mean())}
        res["knn5_label_disagreement"] = float((~same[:, :5]).mean())
        out[name] = res
        out[name]["elapsed_sec"] = round(time.time() - t0, 1)
        r = res["h_median_r5"]
        log(f"twofactor {name}: Delta_bar={r['Delta_bar']:.4f} delta={r['delta_bar_kernel']:.3f} "
            f"iota={r['iota_bar']:.3f} W_mean={r['W_mean']:.1f} tail_w={r['kernel_weight_at_kth_neighbour_mean']:.3f} "
            f"({out[name]['elapsed_sec']}s)")
    save("twofactor", out)


# ───────────────────────────────────────── classes / chance level ──────────

def stage_classes():
    out = {}
    for name in DATASETS:
        _, y = load_xy(name)
        vals, counts = np.unique(y, return_counts=True)
        p = counts / counts.sum()
        raw = load_raw(name).get("label_proxy", {})
        d5 = raw.get("knn5_label_disagreement")
        chance = float(1.0 - (p ** 2).sum())
        H = float(-(p * np.log(p)).sum())
        out[name] = {"n_classes": int(len(vals)), "log_n_classes": float(np.log(len(vals))),
                     "chance_disagreement": chance,
                     "class_entropy_norm": float(H / np.log(len(vals))) if len(vals) > 1 else 1.0,
                     "knn5_label_disagreement": d5,
                     "knn5_disagreement_over_chance": (d5 / chance) if (d5 is not None and chance > 0) else None,
                     "knn5_disagreement_minus_chance": (d5 - chance) if d5 is not None else None}
        log(f"classes {name}: C={len(vals)} chance={chance:.3f} delta5={d5}")
    save("classes", out)


# ───────────────────────────────────────── complexity measures ─────────────

def complexity_measures(X, y):
    n = len(X)
    D = cdist(X, X)
    np.fill_diagonal(D, np.inf)
    nn1 = D.argmin(1)
    n3 = float((y[nn1] != y).mean())                       # N3: LOO 1-NN error
    # N2: intra-class NN distance / inter-class NN distance
    same = (y[:, None] == y[None, :])
    Dintra = np.where(same, D, np.inf); Dinter = np.where(~same, D, np.inf)
    dintra = Dintra.min(1); dinter = Dinter.min(1)
    ok = np.isfinite(dintra) & np.isfinite(dinter)
    n2 = float(dintra[ok].sum() / max(dinter[ok].sum(), 1e-12))
    # N1: fraction of vertices incident to an MST edge that joins different classes
    Dm = D.copy(); np.fill_diagonal(Dm, 0.0)
    T = minimum_spanning_tree(Dm).tocoo()
    boundary = np.zeros(n, dtype=bool)
    diff = y[T.row] != y[T.col]
    boundary[T.row[diff]] = True; boundary[T.col[diff]] = True
    n1 = float(boundary.mean())
    # F1: maximum Fisher discriminant ratio (multiclass form of Lorena et al.)
    classes = np.unique(y)
    mu = X.mean(0)
    num = np.zeros(X.shape[1]); den = np.zeros(X.shape[1])
    for c in classes:
        Xc = X[y == c]; muc = Xc.mean(0)
        num += len(Xc) * (muc - mu) ** 2
        den += ((Xc - muc) ** 2).sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(den > 0, num / den, 0.0)
    f1_ratio = float(ratio.max())
    f1 = float(1.0 / (1.0 + f1_ratio))                      # Lorena's F1 (lower = easier)
    return {"N1": n1, "N2": n2, "N3": n3, "F1": f1, "F1_max_fisher_ratio": f1_ratio}


def stage_complexity():
    out = {}
    for name in DATASETS:
        t0 = time.time()
        X, y = load_xy(name)
        Xs, ys = subsample(X, y, SUB_CPLX)
        res = complexity_measures(Xs.astype(np.float64), ys)
        res["n_used"] = int(len(Xs))
        raw = load_raw(name).get("label_proxy", {})
        res["nn1_error_full_rebuttal"] = raw.get("nn1_error")
        res["elapsed_sec"] = round(time.time() - t0, 1)
        out[name] = res
        log(f"complexity {name}: N1={res['N1']:.3f} N2={res['N2']:.3f} N3={res['N3']:.3f} F1={res['F1']:.3f} ({res['elapsed_sec']}s)")
    save("complexity", out)


# ───────────────────────────────────────── encoding: u-tilde and headline ─

def stage_encoding():
    from experiments.rebuttal_encoding import BUILDERS
    enc_dir = RES / "rebuttal" / "encoding"
    out = {"per_dataset": {}}
    for name in ["adult", "compas", "mushroom", "creditg", "bankmarketing", "nomao"]:
        t0 = time.time()
        builds = BUILDERS[name](ROOT / "data" / "raw")
        rec = {}
        for enc, (X, y) in builds.items():
            nn = NearestNeighbors(n_neighbors=6).fit(X)
            d, _ = nn.kneighbors(X)
            u5 = float(d[:, 5].mean())
            Xp, _ = subsample(X, y, 2000, seed=SEED + 1)
            D = cdist(Xp, Xp)
            med = float(np.median(D[np.triu_indices(len(Xp), 1)]))
            rec[enc] = {"n": int(len(X)), "d": int(X.shape[1]), "u5_mean": u5,
                        "median_pairwise": med, "u_tilde": u5 / med if med > 0 else None}
        j = json.load(open(enc_dir / f"{name}.json"))
        rec["submitted_encoding"] = j["submitted_encoding"]
        rec["dpri_from_rebuttal"] = {e: j["encodings"][e]["dpri"] for e in j["encodings"]}
        rec["n_features_from_rebuttal"] = {e: j["encodings"][e]["n_features"] for e in j["encodings"]}
        out["per_dataset"][name] = rec
        log(f"encoding {name}: " + ", ".join(f"{e}: u={r['u5_mean']:.3f} u~={r['u_tilde']:.3f}" for e, r in rec.items() if isinstance(r, dict) and "u5_mean" in r)
            + f" ({time.time() - t0:.0f}s)")
    save("encoding", out)


# ───────────────────────────────────────── corpus-size curve ───────────────

def stage_corpus(draws=1000, sizes=(7, 10, 15, 20, 25)):
    df = load_features()
    risk = load_risk()
    names = [n for n in DATASETS if n in df.index and n in risk.index]
    df = df.loc[names]
    y = risk.loc[names].values.astype(float)
    cands = [FEATURE_COLS + ["log_nfeatures"], GEO + ["log_nfeatures"], GEO]
    rng = np.random.default_rng(0)
    out = {"draws": draws, "seed": 0, "sizes": {}}
    for m in sizes:
        t0 = time.time()
        rhos = []
        for b in range(draws):
            idx = np.sort(rng.choice(len(names), m, replace=False))
            sub = df.iloc[idx]
            try:
                r = nested_cv_spearman(sub, cands, y[idx])
            except Exception:
                r = float("nan")
            rhos.append(r)
        rhos = np.array(rhos, dtype=float)
        ok = rhos[np.isfinite(rhos)]
        out["sizes"][str(m)] = {"median": float(np.median(ok)), "mean": float(ok.mean()),
                                "p05": float(np.percentile(ok, 5)), "p95": float(np.percentile(ok, 95)),
                                "p25": float(np.percentile(ok, 25)), "p75": float(np.percentile(ok, 75)),
                                "P_neg": float((ok < 0).mean()), "n_ok": int(len(ok))}
        log(f"corpus m={m}: median={out['sizes'][str(m)]['median']:.3f} band=[{out['sizes'][str(m)]['p05']:.3f},{out['sizes'][str(m)]['p95']:.3f}] "
            f"P(<0)={out['sizes'][str(m)]['P_neg']:.3f} ({time.time() - t0:.0f}s)")
        save("corpus", out)   # checkpoint after every size
    save("corpus", out)


# ───────────────────────────────────────── summary ─────────────────────────

def sp(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 4:
        return float("nan"), float("nan"), int(ok.sum())
    r, p = spearmanr(a[ok], b[ok])
    return float(r), float(p), int(ok.sum())


def stage_summary():
    df = load_features(); risk = load_risk()
    names = [n for n in DATASETS if n in df.index and n in risk.index]
    y = risk.loc[names].values.astype(float)
    md = ["# Camera-ready pending checks (#1613) — model-free statistics on the 31-dataset corpus", ""]
    summary = {}

    def add_table(title, rows):
        md.append(f"## {title}"); md.append("")
        md.append("| statistic | Spearman with Risk(D) | p | n |"); md.append("|---|---|---|---|")
        for lab, vals in rows:
            r, p, n = sp(vals, y); summary[lab] = {"rho": r, "p": p, "n": n}
            md.append(f"| {lab} | {r:.3f} | {p:.2e} | {n} |")
        md.append("")

    # W1
    if stage_done("w1"):
        w = json.load(open(OUT / "w1.json"))
        w1 = [w[n]["w1_mean"] for n in names]
        ub_sub = [w[n]["ubar5_subsample"] for n in names]
        uo_sub = [w[n]["uout1_subsample"] for n in names]
        ub_full = df.loc[names, "uniqueness_mean"].values
        r1, _, _ = sp(w1, ub_full); r2, _, _ = sp(w1, ub_sub); r3, _, _ = sp(w1, uo_sub)
        md += ["## (A3') W1 between the member and non-member halves vs the nearest-neighbour spacing", "",
               f"W1 from the optimal matching of {SUB_W1}-point subsamples of each half (3 draws, seed 42 split).", "",
               "| comparison | Spearman |", "|---|---|",
               f"| W1 vs u-bar (paper feature, k=5, full data) | {r1:.3f} |",
               f"| W1 vs u-bar_5 on the same member subsample | {r2:.3f} |",
               f"| W1 vs mean nearest non-member distance (k=1) on the same subsamples | {r3:.3f} |", ""]
        summary["w1"] = {"rho_w1_vs_ubar_full": r1, "rho_w1_vs_ubar_sub": r2, "rho_w1_vs_uout1_sub": r3,
                         "ratio_w1_over_uout1_median": float(np.median(np.array(w1) / np.array(uo_sub)))}
        add_table("W1 as a predictor", [("W1 (3000-point halves)", w1), ("u-bar (paper)", ub_full)])
        md += ["| dataset | W1 | u-bar_5 (subsample) | u_out,1 (subsample) | u-bar (paper) |", "|---|---|---|---|---|"]
        for n, a, b, c, d in zip(names, w1, ub_sub, uo_sub, ub_full):
            md.append(f"| {n} | {a:.3f} | {b:.3f} | {c:.3f} | {d:.3f} |")
        md.append("")

    # two-factor
    if stage_done("twofactor"):
        tf = json.load(open(OUT / "twofactor.json"))
        rows = []
        for tag in ("h_median_r5", "h_median_pairwise", "h_half_median_pairwise"):
            rows.append((f"Delta-bar = mean delta_i*iota_i, Gaussian kernel, {tag}", [tf[n][tag]["Delta_bar"] for n in names]))
            rows.append((f"  kernel-weighted delta-bar, {tag}", [tf[n][tag]["delta_bar_kernel"] for n in names]))
            rows.append((f"  iota-bar, {tag}", [tf[n][tag]["iota_bar"] for n in names]))
        rows.append(("k-NN-5 label disagreement (uniform kernel)", [tf[n]["knn5_label_disagreement"] for n in names]))
        add_table("Proposition 2's own statistic on the corpus", rows)
        md += ["| dataset | h=median r5 | W_mean | tail weight at k=50 | Delta-bar | delta-bar | iota-bar |", "|---|---|---|---|---|---|---|"]
        for n in names:
            r = tf[n]["h_median_r5"]
            md.append(f"| {n} | {r['h']:.3f} | {r['W_mean']:.2f} | {r['kernel_weight_at_kth_neighbour_mean']:.3f} | {r['Delta_bar']:.4f} | {r['delta_bar_kernel']:.3f} | {r['iota_bar']:.3f} |")
        md.append("")

    # classes
    if stage_done("classes"):
        cl = json.load(open(OUT / "classes.json"))
        add_table("Number of classes and chance-normalized label disagreement", [
            ("log C (number of classes)", [cl[n]["log_n_classes"] for n in names]),
            ("chance disagreement 1 - sum p_c^2", [cl[n]["chance_disagreement"] for n in names]),
            ("k-NN-5 label disagreement (rebuttal raw)", [cl[n]["knn5_label_disagreement"] for n in names]),
            ("disagreement / chance", [cl[n]["knn5_disagreement_over_chance"] for n in names]),
            ("disagreement - chance", [cl[n]["knn5_disagreement_minus_chance"] for n in names]),
            ("normalized class entropy (balance)", [cl[n]["class_entropy_norm"] for n in names]),
        ])
        # nested CV with log C offered as a candidate
        d2 = df.loc[names].copy()
        d2["log_nclasses"] = [cl[n]["log_n_classes"] for n in names]
        d2["disagree5"] = [cl[n]["knn5_label_disagreement"] for n in names]
        d2["disagree5_norm"] = [cl[n]["knn5_disagreement_over_chance"] for n in names]
        base = [FEATURE_COLS + ["log_nfeatures"], GEO + ["log_nfeatures"], GEO]
        runs = {
            "paper candidates": base,
            "+ log C offered": base + [c + ["log_nclasses"] for c in base],
            "+ disagreement offered (rebuttal)": base + [c + ["disagree5"] for c in base],
            "+ disagreement + log C offered": base + [c + ["disagree5"] for c in base] + [c + ["disagree5", "log_nclasses"] for c in base],
            "+ chance-normalized disagreement offered": base + [c + ["disagree5_norm"] for c in base],
            "log C alone (LOO)": [["log_nclasses"]],
            "disagreement alone (LOO)": [["disagree5"]],
            "chance-normalized disagreement alone (LOO)": [["disagree5_norm"]],
        }
        if stage_done("complexity"):
            cxj = json.load(open(OUT / "complexity.json"))
            d2["N2"] = [cxj[n]["N2"] for n in names]
            d2["N1"] = [cxj[n]["N1"] for n in names]
            d2["N3"] = [cxj[n]["N3"] for n in names]
            runs.update({
                "+ N2 offered": base + [c + ["N2"] for c in base],
                "+ N2 + disagreement offered": base + [c + ["N2"] for c in base] + [c + ["N2", "disagree5"] for c in base],
                "N2 alone (LOO)": [["N2"]], "N1 alone (LOO)": [["N1"]], "N3 alone (LOO)": [["N3"]],
                "N2 + disagreement (LOO, fixed)": [["N2", "disagree5"]],
            })
        md += ["| nested-CV run | Spearman |", "|---|---|"]
        for lab, cands in runs.items():
            r = nested_cv_spearman(d2, cands, y); summary[f"nested::{lab}"] = float(r)
            md.append(f"| {lab} | {r:.3f} |")
        md.append("")

    # complexity
    if stage_done("complexity"):
        cx = json.load(open(OUT / "complexity.json"))
        add_table("Classical data-complexity measures (5,000-row subsample)", [
            ("N1 (fraction of MST boundary points)", [cx[n]["N1"] for n in names]),
            ("N2 (intra/inter-class NN distance ratio)", [cx[n]["N2"] for n in names]),
            ("N3 (leave-one-out 1-NN error)", [cx[n]["N3"] for n in names]),
            ("F1 (1/(1+max Fisher ratio); lower = easier)", [cx[n]["F1"] for n in names]),
            ("N3 on the full data (rebuttal nn1_error)", [cx[n]["nn1_error_full_rebuttal"] for n in names]),
        ])

    # encoding
    if stage_done("encoding"):
        en = json.load(open(OUT / "encoding.json"))["per_dataset"]
        six = list(en.keys())
        u_int = [en[n]["integer"]["u5_mean"] for n in six]; u_oh = [en[n]["onehot"]["u5_mean"] for n in six]
        ut_int = [en[n]["integer"]["u_tilde"] for n in six]; ut_oh = [en[n]["onehot"]["u_tilde"] for n in six]
        ra_u = spearmanr(u_int, u_oh)[0]; ra_ut = spearmanr(ut_int, ut_oh)[0]
        md += ["## Encoding robustness of the scale-free uniqueness (six re-encoded datasets, 30k subsample)", "",
               f"Rank agreement between integer and one-hot encodings: u {ra_u:.3f} (rebuttal: 0.486); u-tilde {ra_ut:.3f}.", "",
               "| dataset | u int | u one-hot | u~ int | u~ one-hot |", "|---|---|---|---|---|"]
        for n, a, b, c, d in zip(six, u_int, u_oh, ut_int, ut_oh):
            md.append(f"| {n} | {a:.3f} | {b:.3f} | {c:.3f} | {d:.3f} |")
        summary["encoding"] = {"rank_agreement_u": float(ra_u), "rank_agreement_u_tilde": float(ra_ut)}
        # headline with the six datasets' features under the alternative encoding
        cands = [FEATURE_COLS + ["log_nfeatures"], GEO + ["log_nfeatures"], GEO]
        base_rho = nested_cv_spearman(df.loc[names], cands, y)
        variants = {}
        for which in ("submitted", "alternative"):
            d3 = df.loc[names].copy()
            for n in six:
                sub = en[n]["submitted_encoding"]
                enc = sub if which == "submitted" else ("onehot" if sub == "integer" else "integer")
                dp = en[n]["dpri_from_rebuttal"][enc]
                for c in FEATURE_COLS:
                    d3.loc[n, c] = dp[c]
                d3.loc[n, "log_nfeatures"] = np.log(en[n]["n_features_from_rebuttal"][enc])
            variants[which] = float(nested_cv_spearman(d3, cands, y))
        md += ["", "| headline (nested CV, paper candidates) | Spearman |", "|---|---|",
               f"| paper features (full-data run) | {base_rho:.3f} |",
               f"| six datasets replaced by their submitted-encoding features from the 30k re-encoding run | {variants['submitted']:.3f} |",
               f"| six datasets replaced by their alternative-encoding features (30k run) | {variants['alternative']:.3f} |", ""]
        summary["encoding"].update({"headline_base": float(base_rho), **{f"headline_{k}": v for k, v in variants.items()}})

    # corpus
    if stage_done("corpus"):
        co = json.load(open(OUT / "corpus.json"))
        md += [f"## Corpus-size curve ({co['draws']} random subsets per size, seed {co['seed']})", "",
               "| m | median | mean | 5th pct | 95th pct | P(rho<0) |", "|---|---|---|---|---|---|"]
        for m, r in co["sizes"].items():
            md.append(f"| {m} | {r['median']:.3f} | {r['mean']:.3f} | {r['p05']:.3f} | {r['p95']:.3f} | {r['P_neg']:.3f} |")
        md.append("")
        summary["corpus"] = co["sizes"]

    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    (OUT / "SUMMARY.md").write_text("\n".join(md))
    log(f"wrote {OUT / 'SUMMARY.md'}")


STAGES = {"w1": stage_w1, "twofactor": stage_twofactor, "classes": stage_classes,
          "complexity": stage_complexity, "encoding": stage_encoding, "corpus": stage_corpus,
          "summary": stage_summary}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--draws", type=int, default=1000)
    a = ap.parse_args()
    stages = list(STAGES) if a.stage == "all" else a.stage.split(",")
    for s in stages:
        if s != "summary" and stage_done(s) and not a.force:
            log(f"skip {s} (done)"); continue
        log(f"=== stage {s} ===")
        if s == "corpus":
            stage_corpus(draws=a.draws)
        else:
            STAGES[s]()
    if "summary" not in stages:
        stage_summary()


if __name__ == "__main__":
    main()
