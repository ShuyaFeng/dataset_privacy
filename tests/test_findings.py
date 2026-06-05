"""
Tests for Finding 2 (benchmark bias).

Finding 3 (DP calibration) requires opacus + trained models + the processed
datasets, so it is not unit-tested here; it is exercised directly on the
cluster. This file covers the Finding 2 data flow end-to-end, since that is
the paper's headline result.

Run:
    python tests/test_findings.py
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


rf = _load_module("experiments/run_findings.py", "run_findings")


def _make_dpri_csv(td: Path):
    """Synthetic DPRI features: benchmarks deliberately higher than real-world."""
    cols = [
        "uniqueness_mean", "uniqueness_p90", "density_mean", "density_p90",
        "outlier_mean", "outlier_p90", "entropy", "cluster_sep",
    ]
    rows = {
        # benchmarks — high uniqueness, high separation
        "purchase100": [2.0, 3.0, 0.5, 1.0, 0.8, 0.9, 8.0, 0.9],
        "texas100":    [2.2, 3.2, 0.4, 0.9, 0.85, 0.95, 8.5, 0.85],
        # real-world — low uniqueness, low separation
        "adult":     [0.5, 0.8, 5.0, 7.0, 0.2, 0.3, 3.0, 0.1],
        "compas":    [0.6, 0.9, 4.5, 6.5, 0.25, 0.35, 3.2, 0.15],
        "nhanes":    [0.55, 0.85, 4.8, 6.8, 0.22, 0.32, 3.1, 0.12],
        "movielens": [0.7, 1.0, 4.0, 6.0, 0.3, 0.4, 3.5, 0.2],
        "gowalla":   [0.65, 0.95, 4.2, 6.2, 0.28, 0.38, 3.3, 0.18],
    }
    df = pd.DataFrame.from_dict(rows, orient="index", columns=cols)
    df.index.name = "dataset"
    (td / "results" / "dpri").mkdir(parents=True)
    df.to_csv(td / "results" / "dpri" / "dpri_features.csv")


def test_finding2_runs_and_outputs_json():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        _make_dpri_csv(td)
        cwd = os.getcwd()
        try:
            os.chdir(td)
            # OUT_DIR in module is a relative Path; ensure it points into td
            rf.OUT_DIR = Path("results/regression")
            rf.finding2_benchmark_bias()
            out = Path("results/regression/finding2_benchmark_bias.json")
            assert out.exists(), "finding2 JSON not written"
            with open(out) as f:
                results = json.load(f)
        finally:
            os.chdir(cwd)

    # every feature compared
    for feat in ["uniqueness_mean", "density_mean", "outlier_mean", "entropy", "cluster_sep"]:
        assert feat in results, f"missing feature {feat}"
        assert "ks_stat" in results[feat]
        assert "ks_pval" in results[feat]
    print("PASS test_finding2_runs_and_outputs_json")


def test_finding2_detects_benchmark_higher():
    """With benchmarks deliberately higher, the means must reflect that."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        _make_dpri_csv(td)
        cwd = os.getcwd()
        try:
            os.chdir(td)
            rf.OUT_DIR = Path("results/regression")
            rf.finding2_benchmark_bias()
            with open("results/regression/finding2_benchmark_bias.json") as f:
                results = json.load(f)
        finally:
            os.chdir(cwd)

    # uniqueness and cluster_sep were constructed higher for benchmarks
    assert results["uniqueness_mean"]["benchmark_mean"] > results["uniqueness_mean"]["realworld_mean"]
    assert results["cluster_sep"]["benchmark_mean"] > results["cluster_sep"]["realworld_mean"]
    # density was constructed lower for benchmarks
    assert results["density_mean"]["benchmark_mean"] < results["density_mean"]["realworld_mean"]
    print("PASS test_finding2_detects_benchmark_higher")


def test_finding2_ks_stat_valid():
    """KS statistic must be in [0,1] and p-value in [0,1]."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        _make_dpri_csv(td)
        cwd = os.getcwd()
        try:
            os.chdir(td)
            rf.OUT_DIR = Path("results/regression")
            rf.finding2_benchmark_bias()
            with open("results/regression/finding2_benchmark_bias.json") as f:
                results = json.load(f)
        finally:
            os.chdir(cwd)

    for feat, r in results.items():
        assert 0.0 <= r["ks_stat"] <= 1.0, f"{feat} KS stat out of range"
        assert 0.0 <= r["ks_pval"] <= 1.0, f"{feat} p-value out of range"
    print("PASS test_finding2_ks_stat_valid")


if __name__ == "__main__":
    tests = [
        test_finding2_runs_and_outputs_json,
        test_finding2_detects_benchmark_higher,
        test_finding2_ks_stat_valid,
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
