"""
Camera-ready (#1613): per-dataset error bars on Risk(D) and the attenuation
ceiling of the headline, from the attack grid re-run under extra seeds.

Reads results/mia_grid_v2/<dataset>__<attack>__<model>__seed<S>.json for
every seed present (42 = the paper's split; 43, 44 = repeats; the split, the
target models and the shadow models all follow the seed). Writes
results/camera_ready/error_bars.json and results/camera_ready/ERROR_BARS.md.

Quantities:
  * per dataset: Risk_s(D) for each seed, its across-seed SD and range, and the
    per-configuration across-seed SD;
  * reliability: mean pairwise Spearman between the seed-wise Risk vectors
    (test-retest reliability r of the ranking) and the implied ceiling sqrt(r)
    on the Spearman correlation any predictor can reach against a single-seed
    Risk(D); also the intraclass correlation of the dataset means;
  * the headline (nested CV, paper protocol) against the 3-seed mean Risk(D).
For Purchase100 and Texas100 the two XGBoost configurations that were too
expensive to repeat (LiRA and shadow) are held at their seed-42 value when
forming Risk_s (flagged in the output); reliability is reported both over the
29 fully replicated datasets and over all 31.
"""
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from experiments.run_regression import nested_cv_spearman  # noqa: E402

GRID = ROOT / "results" / "mia_grid_v2"
OUT = ROOT / "results" / "camera_ready"
OUT.mkdir(parents=True, exist_ok=True)
FEATURE_COLS = ["uniqueness_mean", "density_mean", "outlier_mean", "entropy", "cluster_sep"]
GEO = ["uniqueness_mean", "density_mean", "cluster_sep"]
ATTACKS = ["loss_threshold", "shadow_model", "lira"]
# the four configurations deliberately not repeated (prohibitive cost); held at seed 42
HELD = {("purchase100", "lira", "xgboost"), ("purchase100", "shadow_model", "xgboost"),
        ("texas100", "lira", "xgboost"), ("texas100", "shadow_model", "xgboost")}
MODELS = ["mlp", "xgboost", "rf"]


def load_grid():
    auc = defaultdict(dict)   # (dataset, attack, model) -> {seed: auc}
    for p in glob.glob(str(GRID / "*.json")):
        d = json.load(open(p))
        auc[(d["dataset"], d["attack"], d["model"])][int(d["seed"])] = float(d["auc"])
    return auc


def main():
    auc = load_grid()
    datasets = sorted({k[0] for k in auc})
    seeds = sorted({s for v in auc.values() for s in v})
    md = ["# Per-dataset error bars on Risk(D) (repeated member/non-member splits)", "",
          f"datasets: {len(datasets)}; seeds present: {seeds}", ""]
    risk = {}        # dataset -> {seed: Risk_s}
    replicated = {}  # dataset -> fully replicated over all seeds?
    cfg_sd = {}
    for d in datasets:
        per_seed, full = {}, True
        sds = []
        for s in seeds:
            vals = []
            for a in ATTACKS:
                for m in MODELS:
                    v = auc.get((d, a, m), {})
                    if s in v:
                        vals.append(v[s])
                    elif (d, a, m) in HELD and 42 in v:
                        vals.append(v[42]); full = False if s != 42 else full
                    else:
                        vals = None; break
                if vals is None: break
            if vals is not None and len(vals) == 9:
                per_seed[s] = float(np.mean(vals))
        for a in ATTACKS:
            for m in MODELS:
                v = auc.get((d, a, m), {})
                if len(v) >= 2: sds.append(float(np.std(list(v.values()), ddof=1)))
        risk[d] = per_seed; replicated[d] = full; cfg_sd[d] = float(np.mean(sds)) if sds else float("nan")
    common = [s for s in seeds if all(s in risk[d] for d in datasets)]
    md.append(f"seeds with a Risk value for every dataset: {common}"); md.append("")
    rows = []
    for d in datasets:
        r = [risk[d][s] for s in common]
        rows.append({"dataset": d, "risk_seed42": risk[d].get(42, float("nan")), "risk_mean": float(np.mean(r)),
                     "risk_sd": float(np.std(r, ddof=1)) if len(r) > 1 else float("nan"),
                     "risk_min": float(min(r)), "risk_max": float(max(r)), "n_seeds": len(r),
                     "mean_config_sd": cfg_sd[d], "fully_replicated": replicated[d]})
    df = pd.DataFrame(rows).set_index("dataset").sort_values("risk_mean", ascending=False)
    md.append("| dataset | Risk seed 42 | mean over seeds | SD | min | max | mean per-config SD | all 9 configs repeated |")
    md.append("|---|---|---|---|---|---|---|---|")
    for d, r in df.iterrows():
        md.append(f"| {d} | {r.risk_seed42:.3f} | {r.risk_mean:.3f} | {r.risk_sd:.3f} | {r.risk_min:.3f} | {r.risk_max:.3f} | {r.mean_config_sd:.3f} | {'yes' if r.fully_replicated else 'no (XGBoost LiRA/shadow held at seed 42)'} |")
    md.append("")
    summ = {"seeds": common, "n_datasets": len(datasets),
            "median_risk_sd": float(df.risk_sd.median()), "max_risk_sd": float(df.risk_sd.max()),
            "argmax_risk_sd": str(df.risk_sd.idxmax()), "median_config_sd": float(df.mean_config_sd.median()),
            "risk_range_seed42": [float(df.risk_seed42.min()), float(df.risk_seed42.max())]}
    md.append(f"Across datasets: median across-seed SD of Risk(D) = {summ['median_risk_sd']:.3f}, max = {summ['max_risk_sd']:.3f} ({summ['argmax_risk_sd']}); "
              f"median per-configuration SD = {summ['median_config_sd']:.3f}; the seed-42 Risk(D) spans {summ['risk_range_seed42'][0]:.2f}-{summ['risk_range_seed42'][1]:.2f}.")
    md.append("")
    # reliability of the ranking
    def reliability(names):
        if len(names) < 3:
            return float('nan'), float('nan'), []
        M = np.array([[risk[d][s] for s in common] for d in names])   # datasets x seeds
        rhos = []
        for i in range(len(common)):
            for j in range(i + 1, len(common)):
                rhos.append(spearmanr(M[:, i], M[:, j])[0])
        r_pair = float(np.mean(rhos)) if rhos else float("nan")
        # ICC(1,1) via one-way ANOVA: datasets as groups, seeds as replicates
        k = M.shape[1]; grand = M.mean(); msb = k * ((M.mean(1) - grand) ** 2).sum() / (M.shape[0] - 1)
        msw = ((M - M.mean(1, keepdims=True)) ** 2).sum() / (M.shape[0] * (k - 1))
        icc = float((msb - msw) / (msb + (k - 1) * msw))
        return r_pair, icc, rhos
    full29 = [d for d in datasets if replicated[d]]
    out = {"per_dataset": df.reset_index().to_dict(orient="records"), "summary": summ}
    if len(common) < 2:
        md.append("Only one seed is complete for every dataset; reliability and error bars need at least two.")
        (OUT / "ERROR_BARS.md").write_text("\n".join(md)); print("\n".join(md)); return
    for label, names in (("all 31 datasets (XGBoost LiRA/shadow of Purchase100 and Texas100 held fixed)", datasets),
                         (f"{len(full29)} fully replicated datasets", full29)):
        r_pair, icc, rhos = reliability(names)
        out[f"reliability::{label}"] = {"mean_pairwise_spearman": r_pair, "pairwise": rhos, "icc11": icc,
                                        "ceiling_sqrt_r": float(np.sqrt(max(r_pair, 0)))}
        md.append(f"Test-retest reliability of the ranking, {label}: mean pairwise Spearman between seed-wise Risk vectors r = {r_pair:.3f} "
                  f"(pairs: {', '.join(f'{x:.3f}' for x in rhos)}); ICC(1,1) of the dataset means = {icc:.3f}; "
                  f"attenuation ceiling sqrt(r) = {np.sqrt(max(r_pair, 0)):.3f} on the Spearman correlation of any predictor with a single-seed Risk(D).")
    md.append("")
    # headline against the seed-averaged target
    feats = pd.read_csv(ROOT / "results" / "dpri" / "dpri_features.csv", index_col=0)
    feats["log_nfeatures"] = np.log(feats["n_features"].clip(lower=1))
    names = [d for d in datasets if d in feats.index]
    cands = [FEATURE_COLS + ["log_nfeatures"], GEO + ["log_nfeatures"], GEO]
    y42 = np.array([risk[d][42] for d in names]); ymean = np.array([np.mean([risk[d][s] for s in common]) for d in names])
    rho42 = float(nested_cv_spearman(feats.loc[names], cands, y42))
    rhomean = float(nested_cv_spearman(feats.loc[names], cands, ymean))
    rho_per_seed = {int(s): float(nested_cv_spearman(feats.loc[names], cands, np.array([risk[d][s] for d in names]))) for s in common}
    out["headline"] = {"nested_cv_vs_seed42": rho42, "nested_cv_vs_seed_mean": rhomean, "nested_cv_per_seed": rho_per_seed,
                       "spearman_seed42_vs_mean_target": float(spearmanr(y42, ymean)[0])}
    md.append(f"Headline (nested CV, paper protocol): {rho42:.3f} against the seed-42 Risk(D) (paper), {rhomean:.3f} against the {len(common)}-seed mean Risk(D); "
              f"per-seed targets: " + ", ".join(f"seed {s}: {v:.3f}" for s, v in rho_per_seed.items()) + ".")
    with open(OUT / "error_bars.json", "w") as f:
        json.dump(out, f, indent=1)
    (OUT / "ERROR_BARS.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
