"""
Camera-ready (#1613): pre-registered external validation corpus.

The two-factor index (geometric DPRI features + the k-NN label-disagreement
rate delta-bar) was constructed after inspecting the residuals of the
31-dataset corpus. This module validates it, and the pre-specified index for
comparison, on a corpus chosen by a fixed rule BEFORE any of its attack
results are seen. The rule and the resulting list are frozen in
external_corpus/CORPUS.json; the git commit that adds that file is the
pre-registration timestamp.

Selection rule (SELECTION_RULE below): every classification dataset of the
OpenML-CC18 benchmark suite (OpenML study 99) that is not already in the
main corpus (same OpenML data_id, or the same dataset under another id),
with at least MIN_N rows and at most MAX_D attributes. No other selection.
Preprocessing is the main corpus's: one-hot encoding of nominal columns,
missing values to 0, subsample to 30k rows, labels integer-coded, features
standardized.

Stages (run in this order; each is idempotent):
  select    fetch CC18 metadata, apply the rule, write external_corpus/
            {CORPUS.json, datasets.txt} and the slurm array scripts
  download  fetch every selected dataset from OpenML -> data/external/<name>.npz
  features  DPRI features (Algorithm 1, k=5) and delta-bar (k-NN label
            disagreement, k=5 and 20) -> results/external/features/<name>.json
            (one dataset per call with --dataset, or all)
  training_table
            build external_corpus/main_corpus_training_table.csv from the
            main corpus's local results (31 rows; committed to the repo so the
            cluster does not need results/)
  evaluate  fit on the 31 (candidate subset chosen by inner LOO, as in the
            paper), predict Risk(D) of the external datasets, report the
            out-of-sample Spearman of the two-factor index and of the
            pre-specified index, with Fisher CI and permutation p, plus
            standalone features, nested CV within the external corpus and on
            the pooled corpus -> results/external/EXTERNAL_VALIDATION.{md,json}

Attack grid: experiments/run_mia_grid.py --data_dir data/external
--out_dir results/external/mia_grid --seed 42 --lira_eval_n 2000 --seed_in_name
(slurm/external_grid_{cpu,gpu}.sh, written by `select`).
"""
import argparse
import json
import re
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ───────────────────────────── pre-registered rule ─────────────────────────
SUITE_ID = 99            # OpenML-CC18
MIN_N = 1000             # rows (before the 30k subsample)
MAX_D = 2000             # attributes as listed by OpenML (target excluded)
MAIN_CORPUS_DIDS = {     # OpenML ids of main-corpus datasets (Table 2 of the paper)
    31, 44, 24, 151, 6, 28, 32, 182, 36, 54, 59, 1489, 1461, 1120, 1486, 1478,
    1476, 554, 40996, 1053, 1067, 15,
    1590,                # adult (main corpus: UCI release)
    37,                  # diabetes = Pima Diabetes (main corpus: UCI release)
}
MAIN_CORPUS_NAMES = {"segment", "mnist_784", "fashion-mnist"}   # same dataset under another id
SELECTION_RULE = (f"OpenML-CC18 (study {SUITE_ID}) classification datasets not in the main corpus "
                  f"(data_id not in {sorted(MAIN_CORPUS_DIDS)} and name not in {sorted(MAIN_CORPUS_NAMES)}), "
                  f"NumberOfInstances >= {MIN_N}, NumberOfFeatures - 1 <= {MAX_D}; all of them, no further selection; "
                  f"ordered by data_id.")
SUBSAMPLE = 30000
SEED = 42

EXT = ROOT / "external_corpus"
DATA = ROOT / "data" / "external"
RES = ROOT / "results" / "external"
FEATURE_COLS = ["uniqueness_mean", "density_mean", "outlier_mean", "entropy", "cluster_sep"]
GEO = ["uniqueness_mean", "density_mean", "cluster_sep"]
ATTACKS = ["loss_threshold", "shadow_model", "lira"]
MODELS = ["mlp", "xgboost", "rf"]


def slug(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())[:20]


def _get_json(url, tries=3):
    for t in range(tries):
        try:
            return json.load(urllib.request.urlopen(url, timeout=120))
        except Exception as e:
            if t == tries - 1:
                raise
            time.sleep(5)


# ───────────────────────────── select ──────────────────────────────────────
def stage_select():
    suite = _get_json(f"https://www.openml.org/api/v1/json/study/{SUITE_ID}")["study"]
    ids = sorted(int(x) for x in suite["data"]["data_id"])
    rows = []
    for i in range(0, len(ids), 40):
        js = _get_json("https://www.openml.org/api/v1/json/data/list/data_id/" + ",".join(map(str, ids[i:i + 40])))
        for d in js["data"]["dataset"]:
            q = {x["name"]: float(x["value"]) for x in d.get("quality", []) if x.get("value") not in (None, "")}
            rows.append({"did": int(d["did"]), "openml_name": d["name"], "n": int(q.get("NumberOfInstances", -1)),
                         "d": int(q.get("NumberOfFeatures", -1)) - 1, "n_classes": int(q.get("NumberOfClasses", -1)),
                         "n_symbolic": int(q.get("NumberOfSymbolicFeatures", -1)),
                         "n_missing": int(q.get("NumberOfMissingValues", -1))})
    rows.sort(key=lambda r: r["did"])
    selected, excluded = [], []
    for r in rows:
        if r["did"] in MAIN_CORPUS_DIDS or r["openml_name"].lower() in MAIN_CORPUS_NAMES:
            why = "in main corpus"
        elif r["n"] < MIN_N:
            why = f"n < {MIN_N}"
        elif r["d"] > MAX_D:
            why = f"d > {MAX_D}"
        else:
            why = None
        if why is None:
            selected.append(dict(r, name=slug(r["openml_name"])))
        else:
            excluded.append(dict(r, reason=why))
    names = [r["name"] for r in selected]
    assert len(names) == len(set(names)), "slug collision"
    EXT.mkdir(exist_ok=True)
    out = {"rule": SELECTION_RULE, "suite": SUITE_ID, "min_n": MIN_N, "max_d": MAX_D, "subsample": SUBSAMPLE,
           "selected_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "n_candidates": len(rows), "n_selected": len(selected), "selected": selected, "excluded": excluded}
    (EXT / "CORPUS.json").write_text(json.dumps(out, indent=1))
    (EXT / "datasets.txt").write_text("\n".join(names) + "\n")
    write_slurm(len(names))
    print(f"CC18 candidates: {len(rows)}; selected: {len(selected)}")
    print(pd.DataFrame(selected)[["did", "openml_name", "name", "n", "d", "n_classes"]].to_string(index=False))


def load_corpus():
    js = json.load(open(EXT / "CORPUS.json"))
    return js["selected"], js


# ───────────────────────────── slurm templates ─────────────────────────────
SLURM_HEAD = """#!/bin/bash
# Generated by `python experiments/external_corpus.py select` for the {n}-dataset
# pre-registered external corpus (external_corpus/datasets.txt). Do not edit by
# hand; re-run `select` to regenerate.
#SBATCH --job-name={job}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time=12:00:00
{gres}#SBATCH --array=0-{last}
#SBATCH --output=logs/{job}_%A_%a.out
#SBATCH --error=logs/{job}_%A_%a.err

set -e
mkdir -p logs
cd "$SLURM_SUBMIT_DIR"
module load Anaconda3 2>/dev/null || module load Miniconda3 2>/dev/null || echo "WARNING: no anaconda module"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dpri
mapfile -t DATASETS < external_corpus/datasets.txt
idx=$SLURM_ARRAY_TASK_ID
"""

SLURM_FEATURES = SLURM_HEAD + """DATASET=${DATASETS[$idx]}
echo "external features task $idx -> $DATASET"
python experiments/external_corpus.py features --dataset "$DATASET"
"""

SLURM_GRID_CPU = SLURM_HEAD + """ATTACKS=(loss_threshold shadow_model lira)
MODELS=(xgboost rf)
DATASET=${DATASETS[$((idx / 6))]}; rem=$((idx % 6))
ATTACK=${ATTACKS[$((rem / 2))]}; MODEL=${MODELS[$((rem % 2))]}
echo "external grid CPU task $idx -> dataset=$DATASET attack=$ATTACK model=$MODEL"
python experiments/run_mia_grid.py --dataset "$DATASET" --attack "$ATTACK" --model "$MODEL" \\
    --data_dir data/external --out_dir results/external/mia_grid --seed 42 --lira_eval_n 2000 --seed_in_name
"""

SLURM_GRID_GPU = SLURM_HEAD + """ATTACKS=(loss_threshold shadow_model lira)
DATASET=${DATASETS[$((idx / 3))]}; ATTACK=${ATTACKS[$((idx % 3))]}
echo "external grid GPU task $idx -> dataset=$DATASET attack=$ATTACK model=mlp"
python experiments/run_mia_grid.py --dataset "$DATASET" --attack "$ATTACK" --model mlp \\
    --data_dir data/external --out_dir results/external/mia_grid --seed 42 --lira_eval_n 2000 --seed_in_name
"""


def _fill(tpl, **kw):
    for k, v in kw.items():                     # plain replacement: bash ${...} in the templates stays intact
        tpl = tpl.replace("{" + k + "}", str(v))
    return tpl


def write_slurm(n):
    sl = ROOT / "slurm"
    (sl / "external_features_array.sh").write_text(_fill(SLURM_FEATURES,
        n=n, job="ext_feats", partition="short", cpus=16, mem="64G", gres="", last=n - 1))
    (sl / "external_grid_cpu.sh").write_text(_fill(SLURM_GRID_CPU,
        n=n, job="ext_grid_cpu", partition="short", cpus=8, mem="32G", gres="", last=6 * n - 1))
    (sl / "external_grid_gpu.sh").write_text(_fill(SLURM_GRID_GPU,
        n=n, job="ext_grid_gpu", partition="pascalnodes", cpus=4, mem="32G", gres="#SBATCH --gres=gpu:1\n", last=3 * n - 1))
    print(f"slurm scripts written for {n} datasets: features 0-{n-1}, grid cpu 0-{6*n-1}, grid gpu 0-{3*n-1}")


# ───────────────────────────── download ────────────────────────────────────
def stage_download(only=None, force=False):
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import StandardScaler
    selected, _ = load_corpus()
    DATA.mkdir(parents=True, exist_ok=True)
    for r in selected:
        name = r["name"]
        if only and name not in only:
            continue
        out = DATA / f"{name}.npz"
        if out.exists() and not force:
            print(f"  exists {name}"); continue
        d = fetch_openml(data_id=r["did"], as_frame=True, parser="auto")
        X = pd.get_dummies(d.data, dummy_na=True)                      # main-corpus preprocessing
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0).values
        X = np.nan_to_num(np.asarray(X, dtype=np.float32))
        _, y = np.unique(np.asarray(d.target).astype(str), return_inverse=True)
        y = y.astype(np.int32)
        if len(X) > SUBSAMPLE:
            idx = np.random.default_rng(SEED).choice(len(X), SUBSAMPLE, replace=False)
            X, y = X[idx], y[idx]
        X = StandardScaler().fit_transform(X).astype(np.float32)
        np.savez_compressed(out, X=X, y=y)
        print(f"  saved {name} (openml {r['did']} {r['openml_name']}): {X.shape}, {len(np.unique(y))} classes")


# ───────────────────────────── features ────────────────────────────────────
def compute_features(X, y, n_jobs=-1):
    from sklearn.neighbors import NearestNeighbors
    from src.dpri.features import compute_dpri_features
    t0 = time.time()
    feats = compute_dpri_features(X, y, k=5)                            # Algorithm 1, as run_dpri.py
    nn = NearestNeighbors(n_neighbors=21, algorithm="auto", n_jobs=n_jobs).fit(X)
    _, idx = nn.kneighbors(X)
    yn = y[idx[:, 1:]]                                                  # neighbour labels, self excluded
    vals, counts = np.unique(y, return_counts=True); p = counts / counts.sum()
    feats.update({
        "disagree5": float((yn[:, :5] != y[:, None]).mean()),          # delta-bar, Eq. (disagreement), k=5
        "disagree20": float((yn[:, :20] != y[:, None]).mean()),
        "nn1_error": float((yn[:, 0] != y).mean()),
        "n_classes": int(len(vals)), "chance_disagreement": float(1.0 - (p ** 2).sum()),
        "n_samples": int(len(X)), "n_features": int(X.shape[1]),
        "elapsed_sec": round(time.time() - t0, 1)})
    return feats


def stage_features(dataset=None, force=False):
    selected, _ = load_corpus()
    out_dir = RES / "features"; out_dir.mkdir(parents=True, exist_ok=True)
    names = [dataset] if dataset else [r["name"] for r in selected]
    for name in names:
        out = out_dir / f"{name}.json"
        if out.exists() and not force:
            print(f"  exists {name}"); continue
        p = DATA / f"{name}.npz"
        if not p.exists():
            print(f"  SKIP {name}: {p} missing (run download)"); continue
        d = np.load(p); X = np.asarray(d["X"], dtype=np.float32); y = np.asarray(d["y"]).astype(int)
        feats = compute_features(X, y); feats["dataset"] = name
        json.dump(feats, open(out, "w"), indent=1)
        print(f"  {name}: u={feats['uniqueness_mean']:.3f} rho={feats['density_mean']:.3f} S={feats['cluster_sep']:.3f} "
              f"delta5={feats['disagree5']:.3f} ({feats['elapsed_sec']}s)")


# ───────────────────────────── training table (main corpus) ────────────────
def stage_training_table():
    """31 rows: DPRI features (results/dpri), delta-bar (results/rebuttal/raw
    label_proxy), Risk(D) = mean AUC of the nine seed-42 configurations
    (results/mia_grid_v2). Written to external_corpus/ and committed."""
    import glob
    feats = pd.read_csv(ROOT / "results" / "dpri" / "dpri_features.csv", index_col=0)
    rows = {}
    for p in glob.glob(str(ROOT / "results" / "mia_grid_v2" / "*.json")):
        d = json.load(open(p))
        if int(d.get("seed", 42)) != 42:
            continue
        rows.setdefault(d["dataset"], {})[(d["attack"], d["model"])] = float(d["auc"])
    tab = []
    for name in feats.index:
        cfg = rows.get(name, {})
        if len(cfg) != 9:
            print(f"  WARNING {name}: {len(cfg)}/9 configurations, skipped"); continue
        raw = json.load(open(ROOT / "results" / "rebuttal" / "raw" / f"{name}_raw.json"))
        r = feats.loc[name].to_dict()
        r.update({"dataset": name, "disagree5": raw["label_proxy"]["knn5_label_disagreement"],
                  "disagree20": raw["label_proxy"]["knn20_label_disagreement"],
                  "risk": float(np.mean(list(cfg.values())))})
        for (a, m), v in cfg.items():
            r[f"auc_{a}_{m}"] = v
        tab.append(r)
    df = pd.DataFrame(tab).set_index("dataset")
    EXT.mkdir(exist_ok=True)
    df.to_csv(EXT / "main_corpus_training_table.csv")
    print(f"training table: {len(df)} datasets -> {EXT / 'main_corpus_training_table.csv'}")


# ───────────────────────────── evaluate ────────────────────────────────────
def _fisher_ci(r, n, z=1.959964):
    if n < 4 or not np.isfinite(r):
        return (float("nan"), float("nan"))
    r = float(np.clip(r, -0.999999, 0.999999)); fz = np.arctanh(r); se = 1.0 / np.sqrt(n - 3)
    return (float(np.tanh(fz - z * se)), float(np.tanh(fz + z * se)))


def _select_by_loo(F, cands, y):
    """The paper's inner loop on the full training corpus: the candidate subset
    with the best leave-one-out Spearman."""
    from scipy.stats import spearmanr
    from sklearn.model_selection import LeaveOneOut
    from experiments.run_regression import _rank_ridge_predict
    best, bs = None, -2.0
    for cols in cands:
        A = F[cols].values; iy, ip = [], []
        for tr, te in LeaveOneOut().split(A):
            ip.append(_rank_ridge_predict(A[tr], A[te], y[tr])[0]); iy.append(y[te[0]])
        s = spearmanr(iy, ip)[0]
        if s > bs:
            bs, best = s, cols
    return best, float(bs)


def stage_evaluate(min_configs=9, n_perm=10000, training_table=None):
    from scipy.stats import spearmanr
    from experiments.run_regression import nested_cv_spearman, _rank_ridge_predict
    selected, corpus = load_corpus()
    train = pd.read_csv(training_table or (EXT / "main_corpus_training_table.csv"), index_col=0)
    train["log_nfeatures"] = np.log(train["n_features"].clip(lower=1))
    # external table
    rows, status = [], []
    for r in selected:
        name = r["name"]; fp = RES / "features" / f"{name}.json"
        cfg = {}
        for a in ATTACKS:
            for m in MODELS:
                p = RES / "mia_grid" / f"{name}__{a}__{m}__seed42.json"
                if p.exists():
                    cfg[(a, m)] = float(json.load(open(p))["auc"])
        status.append({"dataset": name, "features": fp.exists(), "configs": len(cfg)})
        if not fp.exists() or len(cfg) < min_configs:
            continue
        f = json.load(open(fp)); f["risk"] = float(np.mean(list(cfg.values()))); f["n_configs"] = len(cfg)
        for (a, m), v in cfg.items():
            f[f"auc_{a}_{m}"] = v
        rows.append(f)
    st = pd.DataFrame(status)
    md = ["# External validation on the pre-registered corpus", "",
          f"Rule: {corpus['rule']}", f"Selected at (UTC): {corpus['selected_at_utc']}; {corpus['n_selected']} datasets.", "",
          f"Datasets with features and >= {min_configs}/9 attack configurations: {len(rows)} of {len(selected)} "
          f"(features done: {int(st.features.sum())}; grid cells done: {int(st.configs.sum())}/{9*len(selected)}).", ""]
    out = {"rule": corpus["rule"], "n_selected": corpus["n_selected"], "status": status}
    if len(rows) < 4:
        md.append("Fewer than four external datasets are complete; nothing to evaluate yet.")
        RES.mkdir(parents=True, exist_ok=True); (RES / "EXTERNAL_VALIDATION.md").write_text("\n".join(md))
        json.dump(out, open(RES / "external_validation.json", "w"), indent=1); print("\n".join(md)); return
    ext = pd.DataFrame(rows).set_index("dataset")
    ext["log_nfeatures"] = np.log(ext["n_features"].clip(lower=1))
    y_tr = train["risk"].values; y_ext = ext["risk"].values; n = len(ext)
    base = [FEATURE_COLS + ["log_nfeatures"], GEO + ["log_nfeatures"], GEO]
    families = {"pre-specified index (paper candidates)": base,
                "two-factor index (candidates + delta-bar offered)": base + [c + ["disagree5"] for c in base]}
    rng = np.random.default_rng(0)
    out["oos"] = {}
    md += ["## Out-of-sample (fit on the 31 main-corpus datasets, predict the external corpus)", "",
           "| index | subset chosen on the 31 (LOO rho) | Spearman on external | 95% CI (Fisher) | permutation p |", "|---|---|---|---|---|"]
    preds = {}
    for label, cands in families.items():
        cols, loo = _select_by_loo(train, cands, y_tr)
        yp = _rank_ridge_predict(train[cols].values, ext[cols].values, y_tr)
        rho = float(spearmanr(y_ext, yp)[0]); ci = _fisher_ci(rho, n)
        null = np.array([spearmanr(rng.permutation(y_ext), yp)[0] for _ in range(n_perm)])
        p = float((np.sum(null >= rho) + 1) / (n_perm + 1))
        preds[label] = yp
        out["oos"][label] = {"subset": cols, "loo_rho_on_train": loo, "spearman": rho, "fisher_ci": ci, "perm_p": p, "n": n,
                             "predictions": dict(zip(ext.index, map(float, yp)))}
        md.append(f"| {label} | {cols} ({loo:.3f}) | {rho:.3f} | [{ci[0]:.2f}, {ci[1]:.2f}] | {p:.4f} |")
    # fixed two-factor set regardless of selection (pre-specified core + delta-bar)
    cols = GEO + ["log_nfeatures", "disagree5"]
    yp = _rank_ridge_predict(train[cols].values, ext[cols].values, y_tr); rho = float(spearmanr(y_ext, yp)[0])
    out["oos"]["fixed core + log d + delta-bar"] = {"subset": cols, "spearman": rho, "fisher_ci": _fisher_ci(rho, n), "n": n}
    md.append(f"| fixed {{u, rho, S, log d, delta-bar}} (no selection) | {cols} | {rho:.3f} | [{_fisher_ci(rho, n)[0]:.2f}, {_fisher_ci(rho, n)[1]:.2f}] | |")
    md.append("")
    # standalone features on the external corpus
    md += ["## Standalone Spearman with Risk(D) on the external corpus", "", "| feature | external | main corpus |", "|---|---|---|"]
    out["standalone"] = {}
    for c in FEATURE_COLS + ["log_nfeatures", "disagree5", "disagree20", "nn1_error"]:
        if c in train.columns and c in ext.columns:
            re_, rm = float(spearmanr(ext[c], y_ext)[0]), float(spearmanr(train[c], y_tr)[0])
            out["standalone"][c] = {"external": re_, "main": rm}; md.append(f"| {c} | {re_:.3f} | {rm:.3f} |")
    md.append("")
    # nested CV within the external corpus and on the pooled corpus
    md += ["## Nested CV (paper protocol) within the external corpus and on the pooled corpus", "",
           "| candidates | external only | pooled (31 + external) |", "|---|---|---|"]
    out["nested"] = {}
    pooled = pd.concat([train, ext], axis=0, join="inner")
    for label, cands in families.items():
        r_ext = float(nested_cv_spearman(ext, cands, y_ext)); r_pool = float(nested_cv_spearman(pooled, cands, pooled["risk"].values))
        out["nested"][label] = {"external": r_ext, "pooled": r_pool}; md.append(f"| {label} | {r_ext:.3f} | {r_pool:.3f} |")
    md += ["", "## External corpus: features and Risk(D)", "",
           "| dataset | n | d | C | u | rho | S | delta5 | Risk(D) | pred (two-factor) | pred (pre-specified) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    tf, ps = list(families)[1], list(families)[0]
    for i, (name, r) in enumerate(ext.sort_values("risk", ascending=False).iterrows()):
        j = list(ext.index).index(name)
        md.append(f"| {name} | {int(r.n_samples)} | {int(r.n_features)} | {int(r.n_classes)} | {r.uniqueness_mean:.3f} | {r.density_mean:.3f} | "
                  f"{r.cluster_sep:.3f} | {r.disagree5:.3f} | {r.risk:.3f} | {preds[tf][j]:.3f} | {preds[ps][j]:.3f} |")
    out["external_table"] = ext.reset_index().to_dict(orient="records")
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "EXTERNAL_VALIDATION.md").write_text("\n".join(md))
    json.dump(out, open(RES / "external_validation.json", "w"), indent=1)
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["select", "download", "features", "training_table", "evaluate"])
    ap.add_argument("--dataset", default=None, help="features: one dataset (slurm array); download: comma-separated subset")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--min_configs", type=int, default=9, help="evaluate: configurations required per dataset (9 = complete grid)")
    ap.add_argument("--n_perm", type=int, default=10000)
    ap.add_argument("--training_table", default=None)
    a = ap.parse_args()
    if a.stage == "select":
        stage_select()
    elif a.stage == "download":
        stage_download(only=set(a.dataset.split(",")) if a.dataset else None, force=a.force)
    elif a.stage == "features":
        stage_features(dataset=a.dataset, force=a.force)
    elif a.stage == "training_table":
        stage_training_table()
    else:
        stage_evaluate(min_configs=a.min_configs, n_perm=a.n_perm, training_table=a.training_table)


if __name__ == "__main__":
    main()
