"""
Rebuttal (#1613, S&P 2027 Cycle 1): aggregate analyses.  CANONICAL AGGREGATOR.

Reads whatever has been produced so far and skips what is missing. It accepts
the outputs of BOTH per-dataset producers in this repo:
    results/dpri/dpri_features.csv              run_dpri.py --merge
    results/mia_grid_v2/*.json  (preferred)      run_mia_grid.py via slurm/rebuttal_grid_v2_*.sh (TPR@FPR, acc)
    results/mia_grid/*.json     (fallback)       run_mia_grid.py via slurm/mia_*_array.sh
    results/rebuttal/raw/*_raw.json              rebuttal_raw_features.py      (peer producer)
    results/rebuttal/features/*.json             rebuttal_features.py          (this producer)
    results/rebuttal/tpr_at_fpr/*.json           run_mia_tpr_at_fpr.py         optional
    results/rebuttal/dp/*.json                   rebuttal_dp.py                optional
    results/rebuttal/encoding/*.json             rebuttal_encoding.py          optional
    results/rebuttal/recipe/*.json               rebuttal_benchmark_recipe.py  optional
    results/rebuttal/subsample_sweep.json        rebuttal_subsample_sweep.py   optional

Writes (identical content in two places so both READMEs stay valid):
    results/rebuttal/rebuttal_summary.md   and   results/rebuttal/analysis/REBUTTAL_NUMBERS.md
    results/rebuttal/rebuttal_results.json and   results/rebuttal/analysis/*.json
    results/rebuttal/figures/*.png|pdf     and   results/rebuttal/analysis/figures/*.png

Usage:
    python experiments/rebuttal_experiments.py --all            (== --experiment all)
    python experiments/rebuttal_experiments.py --only ci,robustness
    python experiments/rebuttal_experiments.py --all --fast

Protocol: every "nested-CV rho" is produced by the same nested leave-one-
dataset-out procedure as the paper (in-fold rank transform, ridge alpha=1,
inner LOO chooses the candidate subset on the 30 training datasets). The
re-implementation is cross-checked against experiments/run_regression.py's
nested_cv_spearman at start-up (analysis/nested.json).
"""

import argparse
import importlib.util
import json
import math
import shutil
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata, t as tdist
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
RES = Path("results")                      # relative to the working directory (cluster + tests)
OUT = RES / "rebuttal"
ANA = OUT / "analysis"
FIG = OUT / "figures"
FIG2 = ANA / "figures"

FEATURE_COLS = ["uniqueness_mean", "density_mean", "outlier_mean", "entropy", "cluster_sep"]
GEO = ["uniqueness_mean", "density_mean", "cluster_sep"]
ALL6 = FEATURE_COLS + ["log_nfeatures"]
CANDIDATES_PAPER = [ALL6, GEO + ["log_nfeatures"], GEO]
MAIN_SEED = 42

MD = []
RESULTS = {}


def md(line=""):
    MD.append(line)


def fmt_cell(c, fmt):
    if isinstance(c, (float, np.floating)):
        return "nan" if not np.isfinite(c) else fmt.format(c)
    return str(c)


def table(headers, rows, fmt="{:.3f}"):
    md("| " + " | ".join(headers) + " |")
    md("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        md("| " + " | ".join(fmt_cell(c, fmt) for c in r) + " |")
    md()


# ───────────────────────────── statistics helpers ─────────────────────────────

def fisher_ci(r, n):
    if n < 4 or not np.isfinite(r) or abs(r) >= 1:
        return (float("nan"), float("nan"))
    z, se = np.arctanh(r), 1.0 / math.sqrt(n - 3)
    return (float(np.tanh(z - 1.959964 * se)), float(np.tanh(z + 1.959964 * se)))


def bonett_wright_ci(r, n):
    """Bonett & Wright (2000) interval for Spearman's rho."""
    if n < 4 or not np.isfinite(r) or abs(r) >= 1:
        return (float("nan"), float("nan"))
    z, se = np.arctanh(r), math.sqrt((1 + r * r / 2) / (n - 3))
    return (float(np.tanh(z - 1.959964 * se)), float(np.tanh(z + 1.959964 * se)))


def rho_pvalue(r, n):
    if n < 3 or not np.isfinite(r) or abs(r) >= 1:
        return float("nan")
    tt = r * math.sqrt((n - 2) / (1 - r * r))
    return float(2 * tdist.sf(abs(tt), n - 2))


def _rank_ridge_predict(F_tr, F_te, y_tr, alpha=1.0):
    """Verbatim from experiments/run_regression.py."""
    Xt = np.empty_like(F_tr, dtype=float)
    Xv = np.empty_like(F_te, dtype=float)
    for j in range(F_tr.shape[1]):
        col = F_tr[:, j]
        Xt[:, j] = rankdata(col)
        Xv[:, j] = [(col < v).sum() + 0.5 * (col == v).sum() for v in F_te[:, j]]
    scaler = StandardScaler()
    Xt = scaler.fit_transform(Xt)
    Xv = scaler.transform(Xv)
    m = Ridge(alpha=alpha)
    m.fit(Xt, y_tr)
    return m.predict(Xv)


def loo_predict(F, y):
    F, y = np.asarray(F, float), np.asarray(y, float)
    yp = np.empty_like(y)
    for tr, te in LeaveOneOut().split(F):
        yp[te[0]] = _rank_ridge_predict(F[tr], F[te], y[tr])[0]
    return yp


def nested_cv(df, candidates, y):
    """Nested LOO exactly as run_regression.nested_cv_spearman, additionally
    returning the predictions and the subset chosen in each outer fold."""
    candidates = [[c for c in cols if c in df.columns] for cols in candidates]
    candidates = [c for c in candidates if c]
    n = len(df)
    y = np.asarray(y, float)
    y_pred = np.empty(n)
    selected = []
    for tr, te in LeaveOneOut().split(np.arange(n)):
        best_cols, best_score = None, -2.0
        for cols in candidates:
            Fm = df.iloc[tr][cols].values.astype(float)
            iy, ip = [], []
            for itr, ite in LeaveOneOut().split(Fm):
                ip.append(_rank_ridge_predict(Fm[itr], Fm[ite], y[tr][itr])[0])
                iy.append(y[tr][ite[0]])
            s = spearmanr(iy, ip)[0]
            s = -2.0 if not np.isfinite(s) else s
            if s > best_score:
                best_score, best_cols = s, cols
        y_pred[te[0]] = _rank_ridge_predict(df.iloc[tr][best_cols].values.astype(float),
                                            df.iloc[te][best_cols].values.astype(float), y[tr])[0]
        selected.append(tuple(best_cols))
    return y, y_pred, selected


def nested_rho(df, candidates, y):
    yt, yp, _ = nested_cv(df, candidates, y)
    r = spearmanr(yt, yp)[0]
    return float(r) if np.isfinite(r) else float("nan")


def paper_code_nested(df, candidates, y):
    """Cross-check: the paper's own implementation (experiments/run_regression.py)."""
    spec = importlib.util.spec_from_file_location("run_regression", ROOT / "experiments" / "run_regression.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cands = [[c for c in cols if c in df.columns] for cols in candidates]
    return float(mod.nested_cv_spearman(df, [c for c in cands if c], np.asarray(y, float)))


# ───────────────────────────────── loaders ─────────────────────────────────────

def is_variant(name):
    return str(name).endswith("_onehot")


def load_dpri():
    p = RES / "dpri" / "dpri_features.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, index_col=0)
    df.index = df.index.astype(str)
    if "n_features" in df.columns:
        df["log_nfeatures"] = np.log(df["n_features"].clip(lower=1))
    return df


def load_grid():
    """Prefer results/mia_grid_v2 (TPR@FPR + accuracies) when it covers at
    least as many seed-42 configs as results/mia_grid. Returns (main, extra)
    where main has one row per (dataset, attack, model) at the main seed and
    extra holds other seeds (for seed variance)."""
    frames = {}
    for sub in ("mia_grid_v2", "mia_grid"):
        files = sorted((RES / sub).glob("*.json"))
        if files:
            frames[sub] = pd.DataFrame([json.load(open(p)) for p in files])
    if not frames:
        return None, None, None
    for df in frames.values():
        if "seed" not in df.columns:
            df["seed"] = MAIN_SEED
        if "acc_gap" in df.columns and "gen_gap" not in df.columns:
            df["gen_gap"] = df["acc_gap"]
        elif "acc_gap" in df.columns:
            df["gen_gap"] = df["gen_gap"].fillna(df["acc_gap"])
    chosen = "mia_grid"
    if "mia_grid_v2" in frames:
        n2 = (frames["mia_grid_v2"]["seed"] == MAIN_SEED).sum()
        n1 = (frames["mia_grid"]["seed"] == MAIN_SEED).sum() if "mia_grid" in frames else 0
        if n2 >= n1:
            chosen = "mia_grid_v2"
    g = frames[chosen]
    main = g[g["seed"] == MAIN_SEED].drop_duplicates(["dataset", "attack", "model"], keep="last")
    extra = g[g["seed"] != MAIN_SEED]
    return main.reset_index(drop=True), extra.reset_index(drop=True), chosen


def _normalize_feat(j):
    """Map either producer's per-dataset JSON onto one schema:
       k_sweep[k] -> u_mean, rho_mean, purity_mean(optional)
       floor -> median_r_k, frac_at_floor, frac_exact_duplicate, density_mean_floored, density_mean_capped
       entropy_bins, pca -> u_mean, rho_mean, purity_mean(opt), formula -> g_surrogate_mean, ...
       split -> u_in_mean, u_out_mean, ratio_in_over_out, per_sample_spearman (+ k1 variants)"""
    out = {"dataset": j.get("dataset"), "n_samples": j.get("n_samples"), "n_features": j.get("n_features")}
    if "k_sweep" in j:                                   # this repo's rebuttal_features.py
        out["k_sweep"] = j["k_sweep"]
        f = j.get("floor", {})
        out["floor"] = {"median_r_k": f.get("median_r_k"), "frac_at_floor": f.get("frac_at_floor"),
                        "frac_exact_duplicate": f.get("frac_exact_duplicate"),
                        "density_mean_floored": f.get("density_mean_floored"),
                        "density_mean_capped": f.get("density_mean_capped_1e-6")}
        out["entropy_bins"] = j.get("entropy_bins", {})
        if "pca" in j:
            out["pca"] = j["pca"]
        out["formula"] = j.get("formula", {})
        if "split" in j and "ratio_in_over_out" in j["split"]:
            s = j["split"]
            out["split"] = {"u_in_mean": s["u_in_mean"], "u_out_mean": s["u_out_mean"],
                            "ratio_in_over_out": s["ratio_in_over_out"],
                            "per_sample_spearman": s.get("per_sample_spearman")}
    elif "k_variants" in j:                              # peer's rebuttal_raw_features.py
        lp = j.get("label_proxy", {})
        ks = {}
        for k, v in j["k_variants"].items():
            ks[k] = {"u_mean": v["uniqueness_mean"], "rho_mean": v["density_mean"],
                     "u_p90": v.get("uniqueness_p90"),
                     "u_median": v.get("uniqueness_median"), "rho_median": v.get("density_median"),
                     "rho_p90": v.get("density_p90"),
                     "u_trimmed": v.get("uniqueness_trimmed"), "rho_trimmed": v.get("density_trimmed")}
            if k == "5" and "knn5_label_disagreement" in lp:
                ks[k]["purity_mean"] = 1.0 - lp["knn5_label_disagreement"]
            if k == "20" and "knn20_label_disagreement" in lp:
                ks[k]["purity_mean"] = 1.0 - lp["knn20_label_disagreement"]
        out["k_sweep"] = ks
        out["nn1_error"] = lp.get("nn1_error")
        d = j.get("duplicates", {})
        by = d.get("density_by_floor", {})
        out["floor"] = {"median_r_k": d.get("median_r5_nonzero"),
                        "frac_at_floor": by.get("0.1", {}).get("frac_at_floor"),
                        "frac_exact_duplicate": d.get("exact_duplicate_fraction"),
                        "density_mean_floored": by.get("0.1", {}).get("mean"),
                        "density_mean_capped": by.get("0.0", {}).get("mean"),
                        "density_by_floor": by}
        out["entropy_bins"] = j.get("entropy_bins", {})
        if "pca" in j:
            p = j["pca"]
            out["pca"] = {"u_mean": p["uniqueness_mean"], "rho_mean": p["density_mean"],
                          "cluster_sep": p.get("cluster_sep"), "n_components": p.get("dim"),
                          "purity_mean": (1.0 - p["knn5_label_disagreement"]) if "knn5_label_disagreement" in p else None}
        k5 = j["k_variants"].get("5", {})
        out["formula"] = {"g_surrogate_mean": k5.get("formula_exact_mean"),
                          "g_surrogate_median": k5.get("formula_exact_median")}
        out["normalized"] = j.get("normalized", {})
        if "distances" in j:
            s = j["distances"]
            out["split"] = {"u_in_mean": s["u_in_mean_k5"], "u_out_mean": s["u_out_mean_k5"],
                            "ratio_in_over_out": s["ratio_mean_k5"], "per_sample_spearman": s.get("per_sample_spearman_k5"),
                            "ratio_k1": s.get("ratio_mean_k1"), "per_sample_spearman_k1": s.get("per_sample_spearman_k1")}
    return out


def load_feats():
    out = {}
    for sub, suffix in (("features", ".json"), ("raw", "_raw.json")):
        d = OUT / sub
        if not d.exists():
            continue
        for p in sorted(d.glob("*" + suffix)):
            try:
                j = json.load(open(p))
            except Exception:
                continue
            name = j.get("dataset") or p.name.replace(suffix, "")
            if name in out:
                continue                      # first producer found wins
            out[name] = _normalize_feat(j)
    return out


def load_json_dir(sub, skip_suffix="summary.json"):
    d = OUT / sub
    if not d.exists():
        return {}
    out = {}
    for p in sorted(d.glob("*.json")):
        if p.name.endswith(skip_suffix):
            continue
        try:
            j = json.load(open(p))
        except Exception:
            continue
        name = j.get("dataset", p.stem)
        out.setdefault(name, []).append(j)
    return out


def load_dp():
    """Normalize DP outputs of rebuttal_dp.py ({inf, '1', ...} with loss_auc) and
    the earlier format ({baseline, eps_1, ...} with auc) onto one schema."""
    raw = load_json_dir("dp")
    out = {}
    for name, recs in raw.items():
        r = recs[0].get("results", {})
        norm = {}
        for key, v in r.items():
            if key in ("inf", "baseline"):
                lab = "inf"
            else:
                lab = key.replace("eps_", "")
            norm[lab] = {"loss_auc": v.get("loss_auc", v.get("auc")),
                         "test_acc": v.get("test_acc"), "train_acc": v.get("train_acc"),
                         "epsilon_spent": v.get("epsilon_spent"), "lira_auc": v.get("lira_auc"),
                         "loss_tpr_at_fpr_01": v.get("loss_tpr_at_fpr_01", v.get("tpr_at_fpr_01"))}
        out[name] = norm
    return out


def risk_series(grid, metric="auc", models=None, attacks=None, agg="mean"):
    g = grid
    if models is not None:
        g = g[g["model"].isin(models)]
    if attacks is not None:
        g = g[g["attack"].isin(attacks)]
    if metric not in g.columns or g.empty:
        return None
    return g.groupby("dataset")[metric].agg(agg)


def align(F, y_series, exclude_variants=True):
    names = [d for d in F.index if d in y_series.index and (not exclude_variants or not is_variant(d))]
    return F.loc[names], y_series.loc[names].values.astype(float), names


def save_plot(fig, stem):
    for d in (FIG, FIG2):
        d.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.png", dpi=160, bbox_inches="tight")
    shutil.copy(FIG / f"{stem}.png", FIG2 / f"{stem}.png")


# ───────────────────────────────── experiments ─────────────────────────────────

def exp_headline(F, y, names):
    yt, yp, sel = nested_cv(F, CANDIDATES_PAPER, y)
    rho = float(spearmanr(yt, yp)[0])
    n = len(yt)
    try:
        paper = paper_code_nested(F, CANDIDATES_PAPER, y)
    except Exception as e:
        paper = float("nan")
        print(f"  paper-code cross-check skipped: {e}")
    res = {"n": n, "nested_spearman": rho, "paper_code_nested": paper,
           "fisher_ci": fisher_ci(rho, n), "bonett_wright_ci": bonett_wright_ci(rho, n), "p_t": rho_pvalue(rho, n),
           "loo_all6_rho": float(spearmanr(y, loo_predict(F[ALL6].values, y))[0]),
           "loo_geo_logd_rho": float(spearmanr(y, loo_predict(F[GEO + ["log_nfeatures"]].values, y))[0]),
           "loo_geo_rho": float(spearmanr(y, loo_predict(F[GEO].values, y))[0]),
           "predictions": {nm: {"measured": float(a), "predicted": float(b)} for nm, a, b in zip(names, yt, yp)},
           "subset_chosen_per_outer_fold": [list(s) for s in sel]}
    md("## Headline: nested CV reproduction (paper protocol)")
    table(["Estimator", "Spearman rho", "95% CI (Fisher z)", "p"],
          [["Nested CV (headline; this script)", rho, f"[{res['fisher_ci'][0]:.3f}, {res['fisher_ci'][1]:.3f}]", f"{res['p_t']:.2e}"],
           ["Nested CV (paper code, run_regression.py)", paper, "", ""],
           ["Ridge, all six (fixed)", res["loo_all6_rho"], "", ""],
           ["Ridge, geometric core + log d (fixed)", res["loo_geo_logd_rho"], "", ""],
           ["Ridge, geometric core (fixed)", res["loo_geo_rho"], "", ""]])
    if np.isfinite(paper) and abs(paper - rho) > 1e-9:
        md(f"**WARNING**: re-implementation ({rho:.6f}) differs from paper code ({paper:.6f}).")
        md()
    return res, yt, yp, sel


def exp_selection(sel):
    cnt = Counter(sel)
    md("## Inner-loop subset selection per outer fold (Reviewer D Q1; Reviewer B)")
    table(["Selected subset", "Folds"], [[" + ".join(k), v] for k, v in cnt.most_common()], fmt="{}")
    return {"+".join(k): v for k, v in cnt.items()}


def exp_ci(F, y, yt, yp, names, extra_grid, n_boot, n_perm, n_sub, seed=0):
    rng = np.random.default_rng(seed)
    n = len(yt)
    rho = float(spearmanr(yt, yp)[0])
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(yt[idx])) < 3:
            continue
        r = spearmanr(yt[idx], yp[idx])[0]
        if np.isfinite(r):
            boots.append(r)
    boots = np.asarray(boots)
    ci_boot = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))) if len(boots) else (np.nan, np.nan)
    t0 = time.time()
    perm = []
    for i in range(n_perm):
        perm.append(nested_rho(F, CANDIDATES_PAPER, rng.permutation(y)))
        if i == 2:
            print(f"    permutation test: ~{(time.time()-t0)/3*n_perm/60:.1f} min for {n_perm} refits", flush=True)
    perm = np.asarray([p for p in perm if np.isfinite(p)])
    p_perm = float((np.sum(perm >= rho) + 1) / (len(perm) + 1)) if len(perm) else float("nan")
    null95 = float(np.percentile(perm, 95)) if len(perm) else float("nan")
    curve = {}
    for m in [7, 10, 15, 20, 25]:
        if m >= n:
            continue
        rs = []
        for _ in range(n_sub):
            idx = np.sort(rng.choice(n, m, replace=False))
            rs.append(nested_rho(F.iloc[idx], CANDIDATES_PAPER, y[idx]))
        rs = np.asarray([r for r in rs if np.isfinite(r)])
        if len(rs):
            curve[m] = {"median": float(np.median(rs)), "p05": float(np.percentile(rs, 5)),
                        "p95": float(np.percentile(rs, 95)), "p_gt_0.9": float(np.mean(rs > 0.9)),
                        "p_lt_0": float(np.mean(rs < 0)), "draws": int(len(rs))}
    jk = {nm: nested_rho(F.iloc[[j for j in range(n) if j != i]], CANDIDATES_PAPER, np.delete(y, i))
          for i, nm in enumerate(names)}
    influence = {}
    drops = {"without Texas100 and Purchase100": [nm for nm in names if nm in ("texas100", "purchase100")],
             "without the 3 most unique datasets": list(F["uniqueness_mean"].sort_values(ascending=False).index[:3]),
             "without the 3 highest-risk datasets": [names[i] for i in np.argsort(-y)[:3]],
             "without the 3 lowest-risk datasets": [names[i] for i in np.argsort(y)[:3]],
             "without datasets with n < 2000": [nm for nm in names if "n_samples" in F.columns and F.loc[nm, "n_samples"] < 2000]}
    for lab, drop in drops.items():
        keep = [i for i, nm in enumerate(names) if nm not in drop]
        if len(keep) >= 10 and len(keep) < n:
            influence[lab] = {"dropped": drop, "rho": nested_rho(F.iloc[keep], CANDIDATES_PAPER, y[keep]), "n": len(keep)}
    jk_vals = np.asarray(list(jk.values()))
    jk_se = float(np.sqrt((n - 1) / n * np.sum((jk_vals - jk_vals.mean()) ** 2)))
    # Risk(D) measurement noise across its configurations and across seeds (if extra seeds exist)
    seed_sd = float("nan")
    if extra_grid is not None and not extra_grid.empty:
        allseeds = pd.concat([extra_grid], ignore_index=True)
        per = allseeds.groupby(["dataset", "attack", "model"])["auc"].std()
        seed_sd = float(per.mean())
    res = {"rho": rho, "fisher_ci": fisher_ci(rho, n), "bonett_wright_ci": bonett_wright_ci(rho, n),
           "p_t": rho_pvalue(rho, n), "bootstrap_pairs_ci": ci_boot, "bootstrap_n": int(len(boots)),
           "permutation_p": p_perm, "permutation_n": int(len(perm)), "permutation_null_95pct": null95,
           "permutation_null_mean": float(perm.mean()) if len(perm) else float("nan"),
           "corpus_size_curve": curve, "jackknife": jk, "jackknife_se": jk_se,
           "jackknife_min": float(jk_vals.min()), "jackknife_max": float(jk_vals.max()),
           "jackknife_most_influential": names[int(np.argmax(np.abs(jk_vals - rho)))],
           "seed_sd_of_auc_per_config": seed_sd, "influence": influence}
    md("## Admin item 4: error bars on the headline")
    table(["Estimate", "Value"],
          [["Point estimate (nested CV)", f"{rho:.3f}"],
           ["95% CI, Fisher z", f"[{res['fisher_ci'][0]:.3f}, {res['fisher_ci'][1]:.3f}]"],
           ["95% CI, Bonett-Wright", f"[{res['bonett_wright_ci'][0]:.3f}, {res['bonett_wright_ci'][1]:.3f}]"],
           [f"95% CI, dataset-pairs bootstrap (B={len(boots)})", f"[{ci_boot[0]:.3f}, {ci_boot[1]:.3f}]"],
           [f"Permutation test of the full nested procedure (B={len(perm)})", f"p = {p_perm:.4f}; null 95th pct = {null95:.3f}"],
           ["Delete-one jackknife: SE; min / max rho; most influential", f"{jk_se:.3f}; {res['jackknife_min']:.3f} / {res['jackknife_max']:.3f}; {res['jackknife_most_influential']}"],
           ["Mean across-seed SD of AUC per configuration (extra seeds)", f"{seed_sd:.4f}"],
           ["t-test p", f"{res['p_t']:.2e}"]], fmt="{}")
    if influence:
        md("Influence of dataset groups (nested CV refit without them):")
        table(["Subset removed", "n left", "nested-CV rho", "datasets removed"],
              [[lab, v["n"], v["rho"], ", ".join(v["dropped"])] for lab, v in influence.items()], fmt="{:.3f}")
    md("Corpus-size curve (nested CV refit on random subsets of datasets):")
    table(["m", "median rho", "5th pct", "95th pct", "P(rho>0.9)", "P(rho<0)"],
          [[m, c["median"], c["p05"], c["p95"], c["p_gt_0.9"], c["p_lt_0"]] for m, c in curve.items()])
    _plot_ci(boots, perm, rho, curve, n)
    return res


def _plot_ci(boots, perm, rho, curve, n):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(8, 3))
        if len(boots):
            axes[0].hist(boots, bins=40, alpha=0.7, label="pairs bootstrap")
        if len(perm):
            axes[0].hist(perm, bins=40, alpha=0.6, label="permutation null (refit)")
        axes[0].axvline(rho, color="k", lw=1.2, label=f"observed ρ = {rho:.2f}")
        axes[0].set_xlabel("Spearman ρ"); axes[0].legend(fontsize=7); axes[0].set_title("Error bars")
        ms = sorted(curve)
        if ms:
            axes[1].fill_between(ms + [n], [curve[m]["p05"] for m in ms] + [rho], [curve[m]["p95"] for m in ms] + [rho], alpha=0.25)
            axes[1].plot(ms + [n], [curve[m]["median"] for m in ms] + [rho], "o-")
        axes[1].axhline(0, color="k", lw=0.8)
        axes[1].set_xlabel("number of datasets"); axes[1].set_ylabel("nested-CV ρ"); axes[1].set_title("Corpus-size curve")
        for ax in axes:
            ax.grid(alpha=0.3)
        fig.tight_layout()
        save_plot(fig, "bootstrap_ci")
        fig2, ax = plt.subplots(figsize=(4.2, 3))
        if ms:
            ax.fill_between(ms + [n], [curve[m]["p05"] for m in ms] + [rho], [curve[m]["p95"] for m in ms] + [rho], alpha=0.25, label="5–95% band")
            ax.plot(ms + [n], [curve[m]["median"] for m in ms] + [rho], "o-", label="median nested-CV ρ")
        ax.axhline(0, color="k", lw=0.8); ax.set_xlabel("number of datasets in the corpus"); ax.set_ylabel("nested-CV Spearman ρ")
        ax.legend(fontsize=7); ax.grid(alpha=0.3); fig2.tight_layout()
        save_plot(fig2, "corpus_size_curve")
        plt.close("all")
    except Exception as e:
        print(f"  CI plots skipped: {e}")


def exp_robustness(F, grid, tpr_hi):
    defs = [
        ("Mean AUC, all nine configurations (paper)", dict(metric="auc")),
        ("Mean AUC, regularized models only (MLP, XGBoost)", dict(metric="auc", models=["mlp", "xgboost"])),
        ("Median AUC over nine configurations", dict(metric="auc", agg="median")),
        ("Max AUC over nine configurations", dict(metric="auc", agg="max")),
        ("Mean AUC, LiRA only", dict(metric="auc", attacks=["lira"])),
        ("Mean AUC, loss-threshold only", dict(metric="auc", attacks=["loss_threshold"])),
        ("Mean AUC, shadow-model only", dict(metric="auc", attacks=["shadow_model"])),
        ("Mean AUC, MLP only", dict(metric="auc", models=["mlp"])),
        ("Mean AUC, XGBoost only", dict(metric="auc", models=["xgboost"])),
        ("Mean AUC, Random Forest only", dict(metric="auc", models=["rf"])),
        ("Mean AUC, LiRA on regularized models", dict(metric="auc", attacks=["lira"], models=["mlp", "xgboost"])),
        ("TPR at low FPR: mean TPR@10%FPR, LiRA", dict(metric="tpr_at_fpr_10", attacks=["lira"])),
        ("TPR at low FPR: mean TPR@1%FPR, LiRA", dict(metric="tpr_at_fpr_01", attacks=["lira"])),
        ("TPR at low FPR: mean TPR@0.1%FPR, LiRA", dict(metric="tpr_at_fpr_001", attacks=["lira"])),
        ("TPR at low FPR: mean TPR@1%FPR, all nine configurations", dict(metric="tpr_at_fpr_01")),
        ("TPR at low FPR: mean TPR@0.1%FPR, all nine configurations", dict(metric="tpr_at_fpr_001")),
        ("TPR at low FPR: mean TPR@1%FPR, regularized models", dict(metric="tpr_at_fpr_01", models=["mlp", "xgboost"])),
    ]
    base = risk_series(grid, "auc")
    rows, res = [], {}
    for label, kw in defs:
        s = risk_series(grid, **kw)
        if s is None:
            continue
        Fa, ya, nm = align(F, s)
        if len(ya) < 10 or np.nanstd(ya) == 0:
            continue
        rho = nested_rho(Fa, CANDIDATES_PAPER, ya)
        ci = fisher_ci(rho, len(ya))
        agree = float(spearmanr(ya, base.loc[nm].values)[0])
        res[label] = {"rho": rho, "ci": ci, "n": int(len(ya)), "mean_metric": float(np.mean(ya)), "spearman_with_paper_risk": agree}
        rows.append([label, int(len(ya)), rho, f"[{ci[0]:.3f}, {ci[1]:.3f}]", float(np.mean(ya)), agree])
    if tpr_hi:
        hi = pd.DataFrame([r for v in tpr_hi.values() for r in v])
        for metric, label in [("auc", "High-res LiRA (eval_n=2000): AUC"),
                              ("tpr_at_fpr_01", "TPR at low FPR: high-res LiRA TPR@1%FPR"),
                              ("tpr_at_fpr_001", "TPR at low FPR: high-res LiRA TPR@0.1%FPR")]:
            if metric not in hi.columns:
                continue
            s = hi.groupby("dataset")[metric].mean()
            Fa, ya, nm = align(F, s)
            if len(ya) < 10 or np.nanstd(ya) == 0:
                continue
            rho = nested_rho(Fa, CANDIDATES_PAPER, ya)
            ci = fisher_ci(rho, len(ya))
            agree = float(spearmanr(ya, base.loc[nm].values)[0])
            res[label] = {"rho": rho, "ci": ci, "n": int(len(ya)), "mean_metric": float(np.mean(ya)), "spearman_with_paper_risk": agree}
            rows.append([label, int(len(ya)), rho, f"[{ci[0]:.3f}, {ci[1]:.3f}]", float(np.mean(ya)), agree])
    md("## Robustness to the definition of Risk(D): ground-truth robustness (Reviewer C: no RF; Reviewer B: TPR at low FPR)")
    table(["Ground truth for Risk(D)", "n", "Nested-CV rho", "Fisher-z 95% CI", "mean of metric", "Spearman with paper Risk(D)"], rows)
    if "lira_eval_n" in grid.columns:
        g = grid[grid["attack"] == "lira"]
        md(f"LiRA scored targets per configuration (min/median): {int(g['lira_eval_n'].min())} / {int(g['lira_eval_n'].median())}; "
           f"non-member scores at 0.1% FPR therefore rest on ~{int(g['lira_eval_n'].min()*8*0.001)}+ samples at minimum.")
        md()
    return res


def exp_anova(grid):
    """Finding 1: three-factor eta^2 (balanced design => main effects orthogonal),
    next to the paper's std-of-group-means decomposition."""
    g31 = grid[~grid["dataset"].map(is_variant)]
    if g31.empty or "auc" not in g31.columns:
        return {}

    def eta2(gg):
        grand = gg["auc"].mean()
        ss_tot = float(((gg["auc"] - grand) ** 2).sum())
        out = {}
        for f in ("dataset", "model", "attack"):
            grp = gg.groupby(f)["auc"]
            ss = float(sum(len(v) * (v.mean() - grand) ** 2 for _, v in grp))
            out[f] = ss / ss_tot if ss_tot > 0 else float("nan")
        out["residual_and_interactions"] = max(0.0, 1.0 - sum(out[f] for f in ("dataset", "model", "attack")))
        return out

    def spread(gg):
        ds, mo, at = (gg.groupby(k)["auc"].mean().std() for k in ("dataset", "model", "attack"))
        tot = ds + mo + at
        return {"dataset": float(ds / tot), "model": float(mo / tot), "attack": float(at / tot)}

    res = {"eta2_all": eta2(g31), "eta2_regularized": eta2(g31[g31["model"] != "rf"]),
           "spread_all": spread(g31), "spread_regularized": spread(g31[g31["model"] != "rf"]),
           "marginal_attack": g31.groupby("attack")["auc"].mean().to_dict(),
           "marginal_model": g31.groupby("model")["auc"].mean().to_dict()}
    md("## Finding 1 decomposition: eta^2 (three-factor ANOVA, main effects) vs the paper's spread-of-group-means")
    table(["Grid", "method", "dataset", "model", "attack", "residual+interactions"],
          [["all nine configs", "eta^2", res["eta2_all"]["dataset"], res["eta2_all"]["model"], res["eta2_all"]["attack"], res["eta2_all"]["residual_and_interactions"]],
           ["regularized only", "eta^2", res["eta2_regularized"]["dataset"], res["eta2_regularized"]["model"], res["eta2_regularized"]["attack"], res["eta2_regularized"]["residual_and_interactions"]],
           ["all nine configs", "spread (paper)", res["spread_all"]["dataset"], res["spread_all"]["model"], res["spread_all"]["attack"], ""],
           ["regularized only", "spread (paper)", res["spread_regularized"]["dataset"], res["spread_regularized"]["model"], res["spread_regularized"]["attack"], ""]])
    table(["Factor", "Level", "Mean AUC (re-run)"],
          [["Attack", a, float(v)] for a, v in res["marginal_attack"].items()] +
          [["Model", m, float(v)] for m, v in res["marginal_model"].items()])
    return res


def exp_formula(F, y, names, feats):
    have = [nm for nm in names if nm in feats and feats[nm].get("formula")]
    if len(have) < 10:
        md("## Admin item 3: theorem formula vs regression — SKIPPED (per-dataset features missing)")
        return {}
    idx = [names.index(nm) for nm in have]
    yy = y[idx]
    Fh = F.loc[have]
    res, rows = {}, []
    # dataset-level plug-in: mean u / mean rho^(1/d)
    d = Fh["n_features"].values.astype(float)
    plug = Fh["uniqueness_mean"].values / np.power(Fh["density_mean"].values, 1.0 / d)
    cands = {"plug-in  u_bar / rho_bar^(1/d)": plug}
    for key, label in [("g_surrogate_mean", "exact per-sample mean, surrogate density"),
                       ("log_g_surrogate_mean", "exact per-sample mean of log g, surrogate density"),
                       ("g_volume_mean", "exact per-sample mean, volume-based density"),
                       ("log_g_volume_mean", "exact per-sample mean of log g, volume-based density")]:
        vals = [feats[nm]["formula"].get(key) for nm in have]
        if all(v is not None and np.isfinite(v) for v in vals):
            cands[label] = np.array(vals, float)
    for label, g in cands.items():
        rho, p = spearmanr(g, yy)
        loo = float(spearmanr(yy, loo_predict(g.reshape(-1, 1), yy))[0])
        two = np.column_stack([g, Fh["cluster_sep"].values])
        three = np.column_stack([g, Fh["cluster_sep"].values, Fh["log_nfeatures"].values])
        loo2 = float(spearmanr(yy, loo_predict(two, yy))[0])
        loo3 = float(spearmanr(yy, loo_predict(three, yy))[0])
        res[label] = {"standalone_rho": float(rho), "p": float(p), "ci": fisher_ci(float(rho), len(have)),
                      "loo_rho": loo, "loo_plus_S": loo2, "loo_plus_S_logd": loo3}
        rows.append([label, float(rho), f"{p:.3g}", loo, loo2, loo3])
    rho_u, p_u = spearmanr(Fh["uniqueness_mean"], yy)
    rho_r, p_r = spearmanr(Fh["density_mean"], yy)
    res["uniqueness_alone"] = {"standalone_rho": float(rho_u), "p": float(p_u)}
    res["density_alone"] = {"standalone_rho": float(rho_r), "p": float(p_r)}
    res["geo_logd_loo"] = float(spearmanr(yy, loo_predict(Fh[GEO + ["log_nfeatures"]].values, yy))[0])
    res["all6_loo"] = float(spearmanr(yy, loo_predict(Fh[ALL6].values, yy))[0])
    res["nested"] = nested_rho(Fh, CANDIDATES_PAPER, yy)
    res["n"] = len(have)
    md("## Admin item 3: theorem formula u/rho^(1/d) as a predictor vs the regression")
    table(["Predictor", "standalone rho", "p", "LOO rho (alone)", "LOO rho (+ S)", "LOO rho (+ S + log d)"], rows)
    table(["Reference", "rho"],
          [["uniqueness alone (standalone)", float(rho_u)], ["density alone (standalone)", float(rho_r)],
           ["geometric core + log d (LOO)", res["geo_logd_loo"]], ["all six (LOO)", res["all6_loo"]],
           ["nested CV (headline, same datasets)", res["nested"]]])
    _plot_formula(cands[next(iter(cands))], yy, have)
    return res


def _plot_formula(g, y, names):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4.2, 3))
        ax.scatter(np.log(np.maximum(g, 1e-12)), y, s=25)
        for gi, yi, nm in zip(g, y, names):
            ax.annotate(nm, (np.log(max(gi, 1e-12)), yi), fontsize=5, xytext=(2, 2), textcoords="offset points")
        ax.set_xlabel("log geometric factor u / rho^(1/d)"); ax.set_ylabel("measured Risk(D)"); ax.grid(alpha=0.3)
        fig.tight_layout(); save_plot(fig, "formula_vs_regression"); plt.close(fig)
    except Exception as e:
        print(f"  formula plot skipped: {e}")


def _gen_gap(grid, models=("mlp", "xgboost")):
    if "gen_gap" not in grid.columns:
        return None
    g = grid[(grid["attack"] == "loss_threshold") & (grid["model"].isin(models))]
    g = g.dropna(subset=["gen_gap"])
    return g.groupby("dataset")["gen_gap"].mean() if not g.empty else None


def exp_failures(F, y, yt, yp, names, grid, feats):
    resid = yt - yp
    rank_resid = rankdata(yt) - rankdata(yp)
    df = pd.DataFrame({"dataset": names, "measured": yt, "predicted": yp, "residual": resid, "rank_residual": rank_resid})
    df["n"] = [int(F.loc[nm, "n_samples"]) if "n_samples" in F.columns else -1 for nm in names]
    df["d"] = [int(F.loc[nm, "n_features"]) if "n_features" in F.columns else -1 for nm in names]
    gg = _gen_gap(grid)
    df["gen_gap"] = [float(gg.get(nm, np.nan)) if gg is not None else np.nan for nm in names]
    df["purity5"] = [feats[nm]["k_sweep"].get("5", {}).get("purity_mean", np.nan) if nm in feats else np.nan for nm in names]
    df["purity5"] = df["purity5"].astype(float)
    df["n_classes"] = [feats[nm].get("n_classes", np.nan) if nm in feats else np.nan for nm in names]
    pur_hi = np.nanpercentile(df["purity5"], 67) if df["purity5"].notna().any() else np.nan
    pur_lo = np.nanpercentile(df["purity5"], 33) if df["purity5"].notna().any() else np.nan
    gap_lo = np.nanpercentile(df["gen_gap"], 33) if df["gen_gap"].notna().any() else np.nan
    modes = []
    for _, r in df.iterrows():
        if abs(r.residual) <= 0.05:
            modes.append("well predicted")
        elif r.residual < 0:
            easy = (np.isfinite(r.purity5) and r.purity5 >= pur_hi) or (np.isfinite(r.gen_gap) and r.gen_gap <= gap_lo)
            modes.append("easy task, no leakage" if easy else "over-predicted, other")
        else:
            if 0 < r.n < 2000:
                modes.append("small-sample uniqueness inflation")
            elif np.isfinite(r.purity5) and r.purity5 <= pur_lo:
                modes.append("noisy labels, unanticipated memorization")
            else:
                modes.append("under-predicted, other")
    df["mode"] = modes
    df = df.reindex(df["residual"].abs().sort_values(ascending=False).index)
    corr = {}
    for col, lab in [("n", "log n"), ("d", "log d"), ("purity5", "k-NN label purity (k=5) = 1 - label disagreement"),
                     ("gen_gap", "generalization gap (post-training)"), ("n_classes", "number of classes")]:
        v = df[col].astype(float).values
        if col in ("n", "d"):
            v = np.log(np.maximum(v, 1))
        ok = np.isfinite(v)
        if ok.sum() >= 5 and np.std(v[ok]) > 0:
            r, p = spearmanr(v[ok], df["residual"].values[ok])
            r2, _ = spearmanr(v[ok], df["measured"].values[ok])
            corr[lab] = {"rho_with_residual": float(r), "p": float(p), "rho_with_risk": float(r2), "n": int(ok.sum())}
    counts = df["mode"].value_counts().to_dict()
    md("## Admin item 1: systematic failure analysis (nested-CV residuals)")
    table(["Dataset", "n", "d", "measured", "predicted", "residual", "rank resid.", "gen gap", "purity k=5", "mode"],
          [[r.dataset, r.n, r.d, r.measured, r.predicted, r.residual, r.rank_residual, r.gen_gap, r.purity5, r["mode"]] for _, r in df.iterrows()])
    table(["Covariate", "Spearman with signed residual", "p", "Spearman with Risk(D)", "n"],
          [[k, v["rho_with_residual"], f"{v['p']:.3g}", v["rho_with_risk"], v["n"]] for k, v in corr.items()])
    md("Failure-mode counts: " + ", ".join(f"{k}: {v}" for k, v in counts.items()))
    md(f"|residual| > 0.05: {int((df['residual'].abs() > 0.05).sum())} of {len(df)}; > 0.10: {int((df['residual'].abs() > 0.10).sum())}; "
       f"|rank residual| >= 5: {int((df['rank_residual'].abs() >= 5).sum())}")
    md()
    _plot_residuals(df)
    return {"table": df.to_dict(orient="records"), "correlations": corr, "mode_counts": counts}


def _plot_residuals(df):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4.6, 3.6))
        markers = {"well predicted": "o", "easy task, no leakage": "v", "small-sample uniqueness inflation": "^",
                   "noisy labels, unanticipated memorization": "D", "over-predicted, other": "x", "under-predicted, other": "+"}
        for mode, mk in markers.items():
            sub = df[df["mode"] == mode]
            if not sub.empty:
                ax.scatter(sub["measured"], sub["predicted"], marker=mk, s=30, label=mode)
        for _, r in df.iterrows():
            ax.annotate(r.dataset, (r.measured, r.predicted), fontsize=5, xytext=(2, 2), textcoords="offset points")
        lo, hi = df[["measured", "predicted"]].min().min() - 0.02, df[["measured", "predicted"]].max().max() + 0.02
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax.set_xlabel("measured Risk(D)"); ax.set_ylabel("nested-CV predicted Risk(D)")
        ax.legend(fontsize=5.5); ax.grid(alpha=0.3); fig.tight_layout()
        save_plot(fig, "failure_residuals"); plt.close(fig)
    except Exception as e:
        print(f"  residual plot skipped: {e}")


def exp_purity(F, y, names, feats):
    have = [nm for nm in names if nm in feats and feats[nm]["k_sweep"].get("5", {}).get("purity_mean") is not None]
    if len(have) < 10:
        md("## Admin item 1 (remedy): k-NN label disagreement proxy — SKIPPED (no per-dataset features)")
        return {}
    idx = [names.index(nm) for nm in have]
    yy = y[idx]
    Fp = F.loc[have].copy()
    Fp["purity5"] = [feats[nm]["k_sweep"]["5"]["purity_mean"] for nm in have]
    Fp["disagree5"] = 1.0 - Fp["purity5"]
    rho_p, p_p = spearmanr(Fp["disagree5"], yy)
    base = nested_rho(Fp, CANDIDATES_PAPER, yy)
    cand = CANDIDATES_PAPER + [GEO + ["log_nfeatures", "disagree5"], ALL6 + ["disagree5"]]
    yt, yp, sel = nested_cv(Fp, cand, yy)
    with_p = float(spearmanr(yt, yp)[0])
    cand_only = [GEO + ["log_nfeatures", "disagree5"], ALL6 + ["disagree5"], GEO + ["disagree5"]]
    only_p = nested_rho(Fp, cand_only, yy)
    n_sel_p = sum(1 for s in sel if "disagree5" in s)
    loo_geo_p = float(spearmanr(yy, loo_predict(Fp[GEO + ["log_nfeatures", "disagree5"]].values, yy))[0])
    res = {"n": len(have), "disagreement_standalone_rho": float(rho_p), "p": float(p_p),
           "nested_without_proxy": base, "nested_with_proxy_candidates_added": with_p,
           "nested_proxy_candidates_only": only_p, "ci_with_proxy": fisher_ci(with_p, len(have)),
           "folds_selecting_proxy": int(n_sel_p), "loo_geo_logd_proxy": loo_geo_p}
    md("## Admin item 1 (remedy): k-NN label disagreement as a pre-training task-difficulty proxy")
    table(["Quantity", "Value"],
          [["k-NN-5 label disagreement: standalone Spearman with Risk(D)", f"{rho_p:.3f} (p={p_p:.3g})"],
           ["nested-CV rho, paper candidate subsets", f"{base:.3f}"],
           ["nested-CV rho, paper candidates + proxy subsets offered to the inner loop", f"{with_p:.3f}  CI [{res['ci_with_proxy'][0]:.3f}, {res['ci_with_proxy'][1]:.3f}]"],
           ["nested-CV rho, proxy subsets only", f"{only_p:.3f}"],
           ["outer folds whose inner loop selected a proxy subset", f"{n_sel_p} / {len(have)}"],
           ["LOO rho, geometric core + log d + proxy (fixed)", f"{loo_geo_p:.3f}"]], fmt="{}")
    return res


def exp_sensitivity(F, y, names, feats):
    have = [nm for nm in names if nm in feats and feats[nm].get("k_sweep")]
    if len(have) < 10:
        md("## Reviewer D: sensitivity — SKIPPED (no per-dataset features)")
        return {}
    idx = [names.index(nm) for nm in have]
    yy = y[idx]
    res = {"k": {}, "bins": {}, "pca": {}, "aggregation": {}, "floor": {}}
    rows = []
    for k in ["3", "5", "10", "20"]:
        Fk = F.loc[have].copy()
        try:
            Fk["uniqueness_mean"] = [feats[nm]["k_sweep"][k]["u_mean"] for nm in have]
            Fk["density_mean"] = [feats[nm]["k_sweep"][k]["rho_mean"] for nm in have]
        except KeyError:
            continue
        pur = [feats[nm]["k_sweep"][k].get("purity_mean") for nm in have]
        ru = spearmanr(Fk["uniqueness_mean"], yy)[0]
        rr = spearmanr(Fk["density_mean"], yy)[0]
        rp = spearmanr(pur, yy)[0] if all(p is not None for p in pur) else float("nan")
        nested = nested_rho(Fk, CANDIDATES_PAPER, yy)
        res["k"][k] = {"u_rho": float(ru), "rho_rho": float(rr), "purity_rho": float(rp), "nested_rho": nested}
        rows.append([k, float(ru), float(rr), float(rp), nested])
    md("## Reviewer D: sensitivity to k, bins, aggregation, density floor, and representation")
    table(["k", "standalone rho(u)", "standalone rho(density)", "standalone rho(purity)", "nested-CV rho"], rows)
    rows = []
    for b in ["10", "20", "50", "100", "200"]:
        try:
            h = np.array([feats[nm]["entropy_bins"][b] for nm in have], float)
        except KeyError:
            continue
        r, p = spearmanr(h, yy)
        res["bins"][b] = {"rho": float(r), "p": float(p)}
        rows.append([b, float(r), f"{p:.3g}"])
    md("Feature entropy vs number of histogram bins:")
    table(["bins", "standalone rho(H)", "p"], rows)
    # aggregation variants (peer producer only)
    rows = []
    for agg in ["median", "p90", "trimmed"]:
        uk = f"u_{agg}"; rk = f"rho_{agg}"
        if all(feats[nm]["k_sweep"].get("5", {}).get(uk) is not None and feats[nm]["k_sweep"].get("5", {}).get(rk) is not None for nm in have):
            Fa = F.loc[have].copy()
            Fa["uniqueness_mean"] = [feats[nm]["k_sweep"]["5"][uk] for nm in have]
            Fa["density_mean"] = [feats[nm]["k_sweep"]["5"][rk] for nm in have]
            nested = nested_rho(Fa, CANDIDATES_PAPER, yy)
            res["aggregation"][agg] = {"u_rho": float(spearmanr(Fa["uniqueness_mean"], yy)[0]),
                                       "rho_rho": float(spearmanr(Fa["density_mean"], yy)[0]), "nested_rho": nested}
            rows.append([agg, res["aggregation"][agg]["u_rho"], res["aggregation"][agg]["rho_rho"], nested])
    if rows:
        md("Aggregation of per-sample u and density (mean is the paper's):")
        table(["aggregation", "standalone rho(u)", "standalone rho(density)", "nested-CV rho"], rows)
    rows = []
    for fl in ["0.0", "0.05", "0.1", "0.2", "0.5"]:
        if all(feats[nm]["floor"].get("density_by_floor", {}).get(fl) for nm in have):
            Ff = F.loc[have].copy()
            Ff["density_mean"] = [min(feats[nm]["floor"]["density_by_floor"][fl]["mean"], 1e12) for nm in have]
            nested = nested_rho(Ff, CANDIDATES_PAPER, yy)
            res["floor"][fl] = {"rho_rho": float(spearmanr(Ff["density_mean"], yy)[0]), "nested_rho": nested}
            rows.append([fl, res["floor"][fl]["rho_rho"], nested])
    if rows:
        md("Density floor (fraction of median r_k; 0.1 is the paper's):")
        table(["floor", "standalone rho(density)", "nested-CV rho"], rows)
    if all("pca" in feats[nm] for nm in have):
        Fp = F.loc[have].copy()
        Fp["uniqueness_mean"] = [feats[nm]["pca"]["u_mean"] for nm in have]
        Fp["density_mean"] = [feats[nm]["pca"]["rho_mean"] for nm in have]
        if all(feats[nm]["pca"].get("cluster_sep") is not None for nm in have):
            Fp["cluster_sep"] = [feats[nm]["pca"]["cluster_sep"] for nm in have]
        nested_pca = nested_rho(Fp, CANDIDATES_PAPER, yy)
        nested_raw = nested_rho(F.loc[have], CANDIDATES_PAPER, yy)
        rank_raw = pd.Series(F.loc[have, "uniqueness_mean"].values, index=have).rank(ascending=False)
        rank_pca = pd.Series(Fp["uniqueness_mean"].values, index=have).rank(ascending=False)
        res["pca"] = {"nested_rho_raw": nested_raw, "nested_rho_pca": nested_pca,
                      "u_rho_raw": float(spearmanr(F.loc[have, "uniqueness_mean"], yy)[0]),
                      "u_rho_pca": float(spearmanr(Fp["uniqueness_mean"], yy)[0]),
                      "ranks": {nm: {"raw": float(rank_raw[nm]), "pca": float(rank_pca[nm])} for nm in have}}
        rows = [["nested-CV rho", nested_raw, nested_pca], ["standalone rho(u)", res["pca"]["u_rho_raw"], res["pca"]["u_rho_pca"]]]
        # normalized uniqueness (peer producer)
        if all(feats[nm].get("normalized", {}).get("uniqueness_over_median_pairwise") for nm in have):
            Fn = F.loc[have].copy()
            Fn["uniqueness_mean"] = [feats[nm]["normalized"]["uniqueness_over_median_pairwise"] for nm in have]
            res["pca"]["u_rho_normalized"] = float(spearmanr(Fn["uniqueness_mean"], yy)[0])
            res["pca"]["nested_rho_normalized"] = nested_rho(Fn, CANDIDATES_PAPER, yy)
            rows.append(["u / median pairwise distance: standalone rho(u) / nested", res["pca"]["u_rho_normalized"], res["pca"]["nested_rho_normalized"]])
        for nm in ["mnist", "fashionmnist", "digits", "optdigits", "texas100", "purchase100"]:
            if nm in have:
                rows.append([f"rank of {nm} by u (1 = most unique): raw / PCA", float(rank_raw[nm]), float(rank_pca[nm])])
        md("PCA-50 representation (Reviewer D, pixel-space distance):")
        table(["Quantity", "raw features", "PCA-50"], rows)
    return res


def exp_distance(feats, names, y=None):
    rows, res = [], {}
    for nm in names:
        s = feats.get(nm, {}).get("split")
        if s and s.get("ratio_in_over_out") is not None:
            res[nm] = s
            rows.append([nm, s["u_in_mean"], s["u_out_mean"], s["ratio_in_over_out"], s.get("per_sample_spearman", float("nan")), s.get("ratio_k1", float("nan"))])
    if not rows:
        md("## Reviewer C: in-sample vs out-of-sample distance — SKIPPED (no per-dataset features)")
        return {}
    ratios = np.array([r[3] for r in rows], float)
    u_in = np.array([r[1] for r in rows], float); u_out = np.array([r[2] for r in rows], float)
    summ = {"mean_ratio": float(ratios.mean()), "median_ratio": float(np.median(ratios)), "min_ratio": float(ratios.min()),
            "max_ratio": float(ratios.max()), "spearman_uin_uout": float(spearmanr(u_in, u_out)[0]),
            "median_per_sample_spearman": float(np.nanmedian([r[4] for r in rows])),
            "n_outside_0.9_1.1": int(np.sum((ratios < 0.9) | (ratios > 1.1))), "n": len(rows)}
    if y is not None:
        yy = np.array([y[names.index(r[0])] for r in rows])
        summ["rho_uin_risk"] = float(spearmanr(u_in, yy)[0]); summ["rho_uout_risk"] = float(spearmanr(u_out, yy)[0])
    md("## Reviewer C: in-sample vs out-of-sample nearest-neighbour distance under the attack-grid split")
    table(["Statistic", "Value"],
          [["mean / median ratio u_in / u_out", f"{summ['mean_ratio']:.4f} / {summ['median_ratio']:.4f} (min {summ['min_ratio']:.3f}, max {summ['max_ratio']:.3f})"],
           ["Spearman(u_in, u_out) across datasets", f"{summ['spearman_uin_uout']:.3f}"],
           ["median within-dataset per-sample Spearman", f"{summ['median_per_sample_spearman']:.3f}"],
           ["datasets with ratio outside [0.9, 1.1]", f"{summ['n_outside_0.9_1.1']} of {summ['n']}"],
           ["Spearman with Risk(D): u_in / u_out", f"{summ.get('rho_uin_risk', float('nan')):.3f} / {summ.get('rho_uout_risk', float('nan')):.3f}"]], fmt="{}")
    table(["Dataset", "mean u_in", "mean u_out", "ratio (k=5)", "per-sample Spearman", "ratio (k=1)"], rows, fmt="{:.4f}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(3.8, 3.4))
        ax.loglog(u_out, u_in, "o", ms=4)
        lo, hi = min(u_in.min(), u_out.min()), max(u_in.max(), u_out.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax.set_xlabel("mean 5-NN distance to non-members (u_out)"); ax.set_ylabel("mean 5-NN distance within members (u_in)")
        ax.grid(alpha=0.3, which="both"); fig.tight_layout(); save_plot(fig, "u_in_vs_u_out"); plt.close(fig)
    except Exception as e:
        print(f"  distance plot skipped: {e}")
    return {"summary": summ, "per_dataset": res}


def exp_density(feats, names):
    rows, res = [], {}
    for nm in names:
        f = feats.get(nm, {}).get("floor")
        if f and f.get("frac_exact_duplicate") is not None:
            res[nm] = f
            rows.append([nm, f.get("median_r_k"), f.get("frac_at_floor"), f.get("frac_exact_duplicate"),
                         f.get("density_mean_floored"), f.get("density_mean_capped")])
    if not rows:
        md("## Reviewer D Q2: Gowalla vs Movielens density — SKIPPED (no per-dataset features)")
        return {}
    rows.sort(key=lambda r: -(r[4] if r[4] is not None else -1))
    md("## Reviewer D Q2: Gowalla vs Movielens density surrogate diagnostics (k=5)")
    focus = [r for r in rows if r[0] in ("movielens", "gowalla")]
    table(["Dataset", "median r_k (non-zero)", "frac at floor", "frac exact duplicates", "mean density (floored, paper)", "mean density (no floor)"],
          focus + [["---"] * 6] + rows, fmt="{:.4g}")
    return res


def exp_guidance(yt, yp, names, dp):
    n = len(yt)
    order_p, order_m = np.argsort(-yp), np.argsort(-yt)
    k_top, k_bot = int(math.ceil(n / 3)), n // 3
    top_p, top_m = set(order_p[:k_top]), set(order_m[:k_top])
    top2_p, bot_p = set(order_p[:n - k_bot]), set(order_p[n - k_bot:])
    mid_p = top2_p - top_p
    missed = [names[i] for i in (top_m - top_p)]
    res = {"n": n, "tercile_size_top": k_top,
           "recall_top_tercile": len(top_p & top_m) / k_top, "precision_top_tercile": len(top_p & top_m) / k_top,
           "recall_top_two_terciles": len(top2_p & top_m) / k_top, "missed_high_risk_datasets": missed,
           "max_measured_in_bottom_tercile": float(max(yt[i] for i in bot_p)),
           "argmax_measured_in_bottom_tercile": names[max(bot_p, key=lambda i: yt[i])],
           "measured_by_pred_tercile": {t: {"n": len(s), "mean": float(np.mean([yt[i] for i in s])), "min": float(min(yt[i] for i in s)), "max": float(max(yt[i] for i in s))}
                                        for t, s in (("top", top_p), ("middle", mid_p), ("bottom", bot_p)) if s}}
    for thr in (0.65, 0.70, 0.75):
        lab = (yt >= thr).astype(int)
        res[f"auc_pred_for_measured_ge_{thr}"] = float(roc_auc_score(lab, yp)) if 0 < lab.sum() < n else float("nan")
    md("## Admin item 2: triage rule evaluated on the corpus (nested-CV predictions)")
    table(["Rule", "Value"],
          [[f"Flag top tercile ({k_top}) -> recall of measured top tercile", f"{100*res['recall_top_tercile']:.0f}%"],
           ["Flag top tercile -> precision", f"{100*res['precision_top_tercile']:.0f}%"],
           ["High-risk datasets missed by the top-tercile flag", ", ".join(missed) if missed else "none"],
           ["Flag top two terciles -> recall of measured top tercile", f"{100*res['recall_top_two_terciles']:.0f}%"],
           ["AUC of predicted score for measured Risk >= 0.65 / 0.70 / 0.75",
            f"{res['auc_pred_for_measured_ge_0.65']:.3f} / {res['auc_pred_for_measured_ge_0.7']:.3f} / {res['auc_pred_for_measured_ge_0.75']:.3f}"],
           ["Highest measured Risk among bottom-tercile (unflagged) datasets", f"{res['max_measured_in_bottom_tercile']:.3f} ({res['argmax_measured_in_bottom_tercile']})"]], fmt="{}")
    table(["Predicted tercile", "n", "mean measured Risk", "min", "max"],
          [[t, v["n"], v["mean"], v["min"], v["max"]] for t, v in res["measured_by_pred_tercile"].items()])
    if dp:
        terc = {names[i]: ("top" if i in top_p else "bottom" if i in bot_p else "middle") for i in range(n)}
        eps_labels, by = None, {}
        for nm, r in dp.items():
            if nm not in terc:
                continue
            if eps_labels is None:
                eps_labels = list(r.keys())
            for e in eps_labels:
                if e in r and r[e].get("loss_auc") is not None:
                    by.setdefault(terc[nm], {}).setdefault(e, []).append(r[e]["loss_auc"])
        if eps_labels:
            rows = []
            for tname in ("top", "middle", "bottom"):
                if tname not in by:
                    continue
                means = {e: float(np.mean(by[tname][e])) for e in eps_labels if e in by[tname]}
                ok = [e for e in eps_labels if e != "inf" and means.get(e, 1) <= 0.55]
                need = max(ok, key=lambda s: float(s)) if ok else "none of the tested budgets"
                rows.append([tname] + [means.get(e, float("nan")) for e in eps_labels] + [need])
                res.setdefault("dp_by_tercile", {})[tname] = {"mean_auc": means, "largest_eps_with_auc_le_0.55": need}
            md("DP-SGD attack AUC (loss-threshold) by predicted tercile; last column = largest tested epsilon that still keeps mean AUC <= 0.55:")
            table(["Predicted tercile"] + [f"eps={e}" for e in eps_labels] + ["largest sufficient eps"], rows)
    return res


def exp_dp(F, y, names, yp, dp):
    have = [nm for nm in names if nm in dp]
    if len(have) < 10:
        md("## Reviewer B: DP-trained models — SKIPPED (fewer than 10 datasets with DP results)")
        return {}
    eps_labels = [e for e in dp[have[0]].keys()]
    idx = [names.index(nm) for nm in have]
    Fh, score, yy = F.loc[have], yp[idx], y[idx]
    base = np.array([dp[nm]["inf"]["loss_auc"] for nm in have]) if "inf" in eps_labels else None
    rows, res = [], {}
    for e in eps_labels:
        auc = np.array([dp[nm][e]["loss_auc"] for nm in have], float)
        acc = np.array([dp[nm][e]["test_acc"] if dp[nm][e]["test_acc"] is not None else np.nan for nm in have], float)
        nested = nested_rho(Fh, CANDIDATES_PAPER, auc) if np.std(auc) > 0 else float("nan")
        ci = fisher_ci(nested, len(have))
        score_rho = float(spearmanr(score, auc)[0]) if np.std(auc) > 0 else float("nan")
        risk_rho = float(spearmanr(yy, auc)[0]) if np.std(auc) > 0 else float("nan")
        drop_rho = float(spearmanr(score, base - auc)[0]) if (base is not None and e != "inf" and np.std(base - auc) > 0) else float("nan")
        lira = [dp[nm][e].get("lira_auc") for nm in have]
        lira_mean = float(np.mean([v for v in lira if v is not None])) if any(v is not None for v in lira) else float("nan")
        res[e] = {"mean_loss_auc": float(auc.mean()), "n_auc_gt_0.55": int((auc > 0.55).sum()), "mean_test_acc": float(np.nanmean(acc)),
                  "nested_rho": nested, "ci": ci, "dpri_score_rho": score_rho, "nonDP_risk_rho": risk_rho,
                  "rho_dpri_vs_auc_drop": drop_rho, "mean_lira_auc": lira_mean}
        rows.append([e, float(auc.mean()), int((auc > 0.55).sum()), float(np.nanmean(acc)), nested, f"[{ci[0]:.3f}, {ci[1]:.3f}]", score_rho, risk_rho, drop_rho, lira_mean])
    md("## Reviewer B: DPRI under DP-trained models (DP-SGD MLP, loss-threshold attack)")
    table(["epsilon", "mean AUC", "# datasets AUC>0.55", "mean test acc", "nested-CV rho (features vs AUC)", "Fisher-z CI",
           "Spearman(DPRI score, AUC)", "Spearman(non-DP Risk(D), AUC)", "Spearman(DPRI score, AUC drop)", "mean LiRA AUC"], rows)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = [e for e in eps_labels if e != "inf"]
        fig, ax = plt.subplots(figsize=(4.2, 3))
        ax.plot([float(e) for e in xs], [res[e]["mean_loss_auc"] for e in xs], "o-", label="mean attack AUC")
        ax.plot([float(e) for e in xs], [res[e]["nested_rho"] for e in xs], "s--", label="nested-CV ρ (DPRI vs AUC)")
        if "inf" in res:
            ax.axhline(res["inf"]["mean_loss_auc"], color="gray", lw=0.8, ls=":", label="AUC without DP")
        ax.set_xscale("log"); ax.set_xlabel("ε"); ax.legend(fontsize=7); ax.grid(alpha=0.3); fig.tight_layout()
        save_plot(fig, "dp_auc_by_tercile"); save_plot(fig, "dp_curves"); plt.close(fig)
    except Exception as e:
        print(f"  dp plot skipped: {e}")
    return res


def exp_encoding(enc, F, y_series):
    """Peer producer rebuttal_encoding.py: both encodings of the categorical datasets."""
    if not enc:
        md("## Reviewer B: encoding sensitivity — SKIPPED (no results/rebuttal/encoding)")
        return {}
    rows, res = [], {}
    u_sub, u_oth, r_sub, r_oth = [], [], [], []
    for nm, recs in enc.items():
        j = recs[0]
        sub = j.get("submitted_encoding")
        oth = "onehot" if sub == "integer" else "integer"
        E = j["encodings"]
        if sub not in E or oth not in E:
            continue
        res[nm] = {"submitted": sub}
        for tag, e in (("submitted", sub), ("other", oth)):
            dp_ = E[e]["dpri"]; mia = E[e]["mia"]
            res[nm][tag] = {"encoding": e, "d": E[e]["n_features"], "u": dp_["uniqueness_mean"], "rho": dp_["density_mean"],
                            "S": dp_.get("cluster_sep"), "mean_auc": mia["mean_auc"]}
            rows.append([nm, e + (" (submitted)" if tag == "submitted" else ""), E[e]["n_features"], dp_["uniqueness_mean"], dp_["density_mean"], dp_.get("cluster_sep"), mia["mean_auc"]])
        u_sub.append(res[nm]["submitted"]["u"]); u_oth.append(res[nm]["other"]["u"])
        r_sub.append(res[nm]["submitted"]["mean_auc"]); r_oth.append(res[nm]["other"]["mean_auc"])
    summ = {}
    if len(u_sub) >= 3:
        summ = {"n_datasets": len(u_sub), "rank_agreement_uniqueness": float(spearmanr(u_sub, u_oth)[0]),
                "rank_agreement_risk": float(spearmanr(r_sub, r_oth)[0]),
                "mean_abs_delta_auc": float(np.mean(np.abs(np.array(r_sub) - np.array(r_oth))))}
    md("## Reviewer B: encoding sensitivity (integer vs one-hot; rebuttal_encoding.py)")
    table(["Dataset", "encoding", "d", "u", "density", "S", "mean AUC (loss-threshold, 3 models)"], rows, fmt="{:.3f}")
    if summ:
        md(f"Across {summ['n_datasets']} datasets: rank agreement between encodings, uniqueness {summ['rank_agreement_uniqueness']:.3f}, "
           f"risk {summ['rank_agreement_risk']:.3f}; mean |ΔAUC| = {summ['mean_abs_delta_auc']:.3f}.")
        md()
    return {"per_dataset": res, "summary": summ}


def exp_recipe(rec):
    """Peer producer rebuttal_benchmark_recipe.py: Shokri-style construction recipe."""
    if not rec:
        md("## Reviewer C: benchmark construction recipe — SKIPPED (no results/rebuttal/recipe)")
        return {}
    rows, res = [], {}
    ratios_u, dS, dA, dA_ctrl, ratios_u_ctrl = [], [], [], [], []
    for nm, recs in rec.items():
        V = recs[0]["variants"]
        if "original" not in V or "step2_cluster_balance" not in V:
            continue
        o, r, c = V["original"], V["step2_cluster_balance"], V.get("subsample_only_control")
        ru = r["geometry"]["uniqueness_mean"] / o["geometry"]["uniqueness_mean"]
        ds = r["geometry"]["cluster_sep"] - o["geometry"]["cluster_sep"]
        da = r["risk"]["mean_auc"] - o["risk"]["mean_auc"]
        res[nm] = {"u_ratio_recipe": ru, "dS_recipe": ds, "dAUC_recipe": da}
        row = [nm, o["geometry"]["n_samples"], r["geometry"]["n_samples"], ru, ds, da]
        if c:
            ruc = c["geometry"]["uniqueness_mean"] / o["geometry"]["uniqueness_mean"]
            dac = c["risk"]["mean_auc"] - o["risk"]["mean_auc"]
            res[nm].update({"u_ratio_control": ruc, "dAUC_control": dac})
            row += [ruc, dac]
            ratios_u_ctrl.append(ruc); dA_ctrl.append(dac)
        else:
            row += [float("nan"), float("nan")]
        rows.append(row); ratios_u.append(ru); dS.append(ds); dA.append(da)
    md("## Reviewer C: benchmark construction recipe applied to ordinary datasets (Corollary 1)")
    table(["Dataset", "n original", "n recipe", "u ratio (recipe/orig)", "ΔS (recipe)", "ΔAUC (recipe)", "u ratio (subsample-only)", "ΔAUC (subsample-only)"], rows)
    summ = {"n": len(rows), "median_u_ratio_recipe": float(np.median(ratios_u)), "median_dS_recipe": float(np.median(dS)),
            "median_dAUC_recipe": float(np.median(dA)), "n_dAUC_positive": int(np.sum(np.array(dA) > 0)),
            "median_u_ratio_control": float(np.median(ratios_u_ctrl)) if ratios_u_ctrl else float("nan"),
            "median_dAUC_control": float(np.median(dA_ctrl)) if dA_ctrl else float("nan")}
    md(f"Medians over {summ['n']} datasets: recipe raises uniqueness x{summ['median_u_ratio_recipe']:.2f}, S by {summ['median_dS_recipe']:+.3f}, "
       f"mean AUC by {summ['median_dAUC_recipe']:+.3f} ({summ['n_dAUC_positive']}/{summ['n']} positive); subsample-only control: uniqueness x{summ['median_u_ratio_control']:.2f}, AUC {summ['median_dAUC_control']:+.3f}.")
    md()
    return {"per_dataset": res, "summary": summ}


def exp_onehot_grid(F, grid, y_series):
    pairs = [("adult", "adult_onehot"), ("compas", "compas_onehot")]
    avail = [(a, b) for a, b in pairs if a in F.index and b in F.index and a in y_series.index and b in y_series.index]
    if not avail:
        return {}
    base_names = [d for d in F.index if not is_variant(d) and d in y_series.index]
    Fb, yb = F.loc[base_names], y_series.loc[base_names].values.astype(float)
    rho_base = nested_rho(Fb, CANDIDATES_PAPER, yb)
    Fs, ys = Fb.copy(), yb.copy()
    for a, b in avail:
        i = base_names.index(a)
        Fs.iloc[i] = F.loc[b]; ys[i] = y_series.loc[b]
    rho_sub = nested_rho(Fs, CANDIDATES_PAPER, ys)
    res = {"nested_rho_integer": rho_base, "nested_rho_onehot_substituted": rho_sub}
    rows = []
    for a, b in avail:
        i = base_names.index(a)
        res[a] = {"risk_int": float(yb[i]), "risk_onehot": float(ys[i]), "u_int": float(F.loc[a, "uniqueness_mean"]), "u_onehot": float(F.loc[b, "uniqueness_mean"]),
                  "rank_measured": (float(rankdata(-yb)[i]), float(rankdata(-ys)[i]))}
        rows.append([a, f"{yb[i]:.3f} -> {ys[i]:.3f}", f"{F.loc[a,'uniqueness_mean']:.2f} -> {F.loc[b,'uniqueness_mean']:.2f}",
                     f"{res[a]['rank_measured'][0]:.0f} -> {res[a]['rank_measured'][1]:.0f}"])
    md("## Reviewer B: one-hot variants through the full 9-config grid (adult_onehot, compas_onehot)")
    table(["Dataset", "Risk(D) int -> one-hot", "u", "measured rank (of 31)"], rows, fmt="{}")
    md(f"Nested-CV rho: integer-coded {rho_base:.3f}; one-hot substituted {rho_sub:.3f}")
    md()
    return res


def exp_subsample():
    p = OUT / "subsample_sweep.json"
    if not p.exists():
        return {}
    j = json.load(open(p))
    md("## Reviewer C: subsampling sweep (rebuttal_subsample_sweep.py)")
    rows = []
    for ds, r in j["per_dataset"].items():
        for a in r["aggregate"]:
            rows.append([ds, r["d"], a["fraction"], a["n"], a["u_mean"], a["rho_mean"],
                         a.get("auc_mlp") if a.get("auc_mlp") is not None else float("nan"),
                         a.get("auc_xgboost") if a.get("auc_xgboost") is not None else float("nan")])
    table(["Dataset", "d", "fraction", "n", "mean u", "mean density", "AUC MLP", "AUC XGB"], rows, fmt="{:.4g}")
    table(["Dataset", "fitted slope of log u vs log n", "predicted -1/d"],
          [[ds, r["slope_u_vs_n"], r["predicted_slope_minus_1_over_d"]] for ds, r in j["per_dataset"].items()], fmt="{:.4f}")
    return {ds: {"slope_u_vs_n": r["slope_u_vs_n"], "predicted": r["predicted_slope_minus_1_over_d"]} for ds, r in j["per_dataset"].items()}


EXPERIMENTS = ["headline", "selection", "ci", "robustness", "anova", "formula", "failures", "purity",
               "sensitivity", "distance", "density", "guidance", "dp", "encoding", "recipe", "onehot", "subsample"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--experiment", default=None, help="'all' or a comma list (alias of --only)")
    ap.add_argument("--only", default="", help="comma list from: " + ",".join(EXPERIMENTS))
    ap.add_argument("--fast", action="store_true", help="fewer bootstrap / permutation / subset draws")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--n_perm", type=int, default=1000)
    ap.add_argument("--n_sub", type=int, default=None)
    ap.add_argument("--n_jobs", type=int, default=-1, help="accepted for compatibility; analyses are single-process")
    args = ap.parse_args()
    sel = args.only or (args.experiment if args.experiment not in (None, "all") else "")
    todo = set(EXPERIMENTS) if not sel else set(sel.split(","))
    if args.fast:
        args.n_boot, args.n_perm = min(args.n_boot, 2000), min(args.n_perm, 100)
    n_sub = args.n_sub if args.n_sub is not None else (200 if args.n_perm >= 200 else max(5, args.n_perm))

    for d in (OUT, ANA, FIG, FIG2):
        d.mkdir(parents=True, exist_ok=True)
    F = load_dpri()
    grid, extra_grid, grid_src = load_grid()
    if F is None or grid is None:
        sys.exit("Need results/dpri/dpri_features.csv and results/mia_grid[_v2]/*.json first.")
    feats = load_feats()
    tpr_hi = load_json_dir("tpr_at_fpr")
    dp = load_dp()
    enc = load_json_dir("encoding")
    rec = load_json_dir("recipe")

    y_all = risk_series(grid, "auc")
    F31, y, names = align(F, y_all)
    print(f"{len(names)} datasets with DPRI features and grid results (grid source: results/{grid_src}); "
          f"{len(feats)} with per-dataset features; {len(dp)} DP; {len(tpr_hi)} high-res LiRA; {len(enc)} encoding; {len(rec)} recipe")
    md("# Rebuttal numbers (#1613) — generated by experiments/rebuttal_experiments.py")
    md(f"datasets: {len(names)}; grid source: results/{grid_src} ({len(grid)} seed-{MAIN_SEED} configs, {0 if extra_grid is None else len(extra_grid)} extra-seed configs); "
       f"per-dataset features: {len(feats)}; DP: {len(dp)}; high-res LiRA: {len(tpr_hi)}; encoding: {len(enc)}; recipe: {len(rec)}")
    md()

    head, yt, yp, selected = exp_headline(F31, y, names)
    RESULTS["nested"] = head
    with open(ANA / "nested.json", "w") as f:
        json.dump(head, f, indent=2)
    if "selection" in todo:
        RESULTS["selection"] = exp_selection(selected)
    if "ci" in todo:
        RESULTS["bootstrap"] = exp_ci(F31, y, yt, yp, names, extra_grid, args.n_boot, args.n_perm, n_sub)
    if "robustness" in todo:
        RESULTS["robustness"] = exp_robustness(F, grid, tpr_hi)
    if "anova" in todo:
        RESULTS["anova"] = exp_anova(grid)
    if "formula" in todo:
        RESULTS["formula"] = exp_formula(F31, y, names, feats)
    if "failures" in todo:
        RESULTS["failures"] = exp_failures(F31, y, yt, yp, names, grid, feats)
    if "purity" in todo:
        RESULTS["labelproxy"] = exp_purity(F31, y, names, feats)
    if "sensitivity" in todo:
        RESULTS["sensitivity"] = exp_sensitivity(F31, y, names, feats)
    if "distance" in todo:
        RESULTS["distances"] = exp_distance(feats, names, y)
    if "density" in todo:
        RESULTS["density_floor"] = exp_density(feats, names)
    if "guidance" in todo:
        RESULTS["triage"] = exp_guidance(yt, yp, names, dp)
    if "dp" in todo:
        RESULTS["dp"] = exp_dp(F31, y, names, yp, dp)
    if "encoding" in todo:
        RESULTS["encoding"] = exp_encoding(enc, F, y_all)
    if "recipe" in todo:
        RESULTS["recipe"] = exp_recipe(rec)
    if "onehot" in todo:
        RESULTS["onehot_grid"] = exp_onehot_grid(F, grid, y_all)
    if "subsample" in todo:
        RESULTS["subsample"] = exp_subsample()

    def _default(o):
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        return str(o)
    with open(OUT / "rebuttal_results.json", "w") as f:
        json.dump(RESULTS, f, indent=2, default=_default)
    for key, val in RESULTS.items():
        with open(ANA / f"{key}.json", "w") as f:
            json.dump(val, f, indent=2, default=_default)
    text = "\n".join(MD) + "\n"
    (OUT / "rebuttal_summary.md").write_text(text)
    (ANA / "REBUTTAL_NUMBERS.md").write_text(text)
    print(f"\nsaved {OUT / 'rebuttal_summary.md'} (== {ANA / 'REBUTTAL_NUMBERS.md'})")


if __name__ == "__main__":
    main()
