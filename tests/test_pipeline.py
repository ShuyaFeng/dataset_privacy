"""
Tests for the regression pipeline and CLI.

Covers:
- run_loo_cv: leave-one-out cross-validation produces sane R² on linear data
- ablation: correctly identifies the feature that drives the target
- end-to-end: synthetic CSVs -> load_data -> regression -> valid output
- cli: dpri_to_scalar and risk_category produce valid outputs

Run:
    python tests/test_pipeline.py
"""

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent


def _load_module(rel_path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rr = _load_module("experiments/run_regression.py", "run_regression")
cli = _load_module("src/dpri/cli.py", "dpri_cli")


# ── run_loo_cv ───────────────────────────────────────────────────────────────

def test_run_loo_cv_recovers_linear_signal():
    """On (near-)linear data, ridge LOO-CV should achieve high R²."""
    rng = np.random.default_rng(0)
    n = 15
    X = rng.normal(size=(n, 8))
    w = np.array([1.0, 0.0, 0.5, 0.0, -0.8, 0.0, 0.3, 0.0])
    y = X @ w + rng.normal(0, 0.01, n)
    r2, spear, yt, yp = rr.run_loo_cv(X, y, "linear")
    assert r2 > 0.7, f"linear signal should give high R², got {r2:.3f}"
    assert -1.0 <= spear <= 1.0
    assert len(yt) == n and len(yp) == n
    print(f"PASS test_run_loo_cv_recovers_linear_signal (R²={r2:.3f})")


def test_run_loo_cv_rf_runs():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(12, 8))
    y = X[:, 0] + rng.normal(0, 0.1, 12)
    r2, spear, yt, yp = rr.run_loo_cv(X, y, "rf")
    assert np.isfinite(r2)
    print(f"PASS test_run_loo_cv_rf_runs (R²={r2:.3f})")


def test_run_loo_cv_rejects_unknown_model():
    X = np.random.default_rng(0).normal(size=(10, 3))
    y = np.random.default_rng(0).normal(size=10)
    try:
        rr.run_loo_cv(X, y, "banana")
        assert False, "should have raised ValueError"
    except ValueError:
        pass
    print("PASS test_run_loo_cv_rejects_unknown_model")


# ── single-factor analysis ───────────────────────────────────────────────────

def test_single_factor_identifies_driving_feature():
    """If Risk depends only on feature 0, it must have the highest |Spearman|."""
    rng = np.random.default_rng(0)
    n = 20
    X = rng.normal(size=(n, 4))
    y = 2.0 * X[:, 0] + rng.normal(0, 0.01, n)
    df = pd.DataFrame(X, columns=["f0", "f1", "f2", "f3"])
    df["Risk_D"] = y
    results = rr.single_factor_analysis(df, ["f0", "f1", "f2", "f3"])
    best = max(results, key=lambda k: abs(results[k]["spearman"]))
    assert best == "f0", f"f0 should dominate, got {best}: {results}"
    assert results["f0"]["p_value"] < 0.05, "f0 should be significant"
    print(f"PASS test_single_factor_identifies_driving_feature (f0 rho={results['f0']['spearman']})")


# ── end-to-end regression from synthetic CSVs ────────────────────────────────

def test_regression_end_to_end():
    """Synthetic dpri + risk CSVs -> load_data -> regression -> valid output."""
    rng = np.random.default_rng(0)
    datasets = [f"ds{i}" for i in range(9)]

    # synthetic DPRI features where risk is a linear function of features
    n_feat = len(rr.FEATURE_COLS)
    feat_data = rng.normal(size=(9, n_feat))
    w = np.linspace(0.3, 0.05, n_feat)
    risk = 0.5 + 0.1 * (feat_data @ w)
    risk = np.clip(risk, 0.4, 1.0)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "results" / "dpri").mkdir(parents=True)
        (td / "results" / "regression").mkdir(parents=True)

        dpri_df = pd.DataFrame(feat_data, columns=rr.FEATURE_COLS, index=datasets)
        dpri_df.index.name = "dataset"
        dpri_df["n_samples"] = 1000
        dpri_df["n_features"] = n_feat
        dpri_df.to_csv(td / "results" / "dpri" / "dpri_features.csv")

        risk_df = pd.DataFrame({"dataset": datasets, "Risk_D": risk})
        risk_df.to_csv(td / "results" / "ground_truth_risk.csv", index=False)

        cwd = os.getcwd()
        try:
            os.chdir(td)
            # run the FULL main() — this also exercises JSON serialization,
            # which is where numpy-type bugs (like the Finding 2 np.bool_) hide
            rr.main()
            out = Path("results/regression/regression_results.json")
            assert out.exists(), "regression_results.json not written"
            with open(out) as f:
                res = json.load(f)
            assert "linear" in res and "r2_loo" in res["linear"]
            assert "rf" in res
            assert "single_factor" in res
            assert "feature_importances" in res
            r2 = res["linear"]["r2_loo"]
            assert np.isfinite(r2)
        finally:
            os.chdir(cwd)
    print(f"PASS test_regression_end_to_end (R²={r2:.3f})")


# ── CLI ──────────────────────────────────────────────────────────────────────

def test_cli_dpri_to_scalar_range():
    feats = {
        "uniqueness_mean": 1.5, "uniqueness_p90": 2.0,
        "density_mean": 3.0, "density_p90": 5.0,
        "outlier_mean": 0.4, "outlier_p90": 0.8,
        "entropy": 4.0, "cluster_sep": 0.3,
    }
    score = cli.dpri_to_scalar(feats)
    assert 0.0 <= score <= 1.0, f"DPRI scalar out of range: {score}"
    print(f"PASS test_cli_dpri_to_scalar_range (score={score:.3f})")


def test_cli_risk_category():
    assert cli.risk_category(0.1) == "Low"
    assert cli.risk_category(0.5) == "Medium"
    assert cli.risk_category(0.9) == "High"
    print("PASS test_cli_risk_category")


def test_cli_higher_features_higher_risk():
    """A dataset with higher uniqueness/outlier should score higher."""
    low = {
        "uniqueness_mean": 0.1, "uniqueness_p90": 0.2,
        "density_mean": 10.0, "density_p90": 15.0,
        "outlier_mean": 0.05, "outlier_p90": 0.1,
        "entropy": 1.0, "cluster_sep": 0.0,
    }
    high = {
        "uniqueness_mean": 2.0, "uniqueness_p90": 3.0,
        "density_mean": 0.5, "density_p90": 1.0,
        "outlier_mean": 0.8, "outlier_p90": 0.95,
        "entropy": 8.0, "cluster_sep": 0.9,
    }
    assert cli.dpri_to_scalar(high) > cli.dpri_to_scalar(low)
    print("PASS test_cli_higher_features_higher_risk")


if __name__ == "__main__":
    tests = [
        test_run_loo_cv_recovers_linear_signal,
        test_run_loo_cv_rf_runs,
        test_run_loo_cv_rejects_unknown_model,
        test_single_factor_identifies_driving_feature,
        test_regression_end_to_end,
        test_cli_dpri_to_scalar_range,
        test_cli_risk_category,
        test_cli_higher_features_higher_risk,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            import traceback
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{'='*50}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
