"""
Unit tests for DPRI feature computation.

These verify the mathematical properties each feature is supposed to capture,
plus edge cases (high dimensionality, duplicate points, small datasets) that
are the most likely cause of run_dpri.py crashing on the cluster.

Run:
    python tests/test_features.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dpri.features import (
    compute_uniqueness,
    compute_density,
    compute_outlier_score,
    compute_entropy,
    compute_cluster_separation,
    compute_dpri_features,
)


# ── Sample Uniqueness ────────────────────────────────────────────────────────

def test_uniqueness_isolated_point_is_highest():
    """An isolated point must have the largest k-NN distance."""
    rng = np.random.default_rng(0)
    cluster = rng.normal(0, 0.1, size=(50, 2))
    isolated = np.array([[100.0, 100.0]])
    X = np.vstack([cluster, isolated])
    u = compute_uniqueness(X, k=5)
    assert np.argmax(u) == len(X) - 1, "isolated point should have max uniqueness"
    print(f"PASS test_uniqueness_isolated_point_is_highest (isolated u={u[-1]:.1f})")


def test_uniqueness_shape():
    X = np.random.default_rng(0).normal(size=(30, 4))
    u = compute_uniqueness(X, k=5)
    assert u.shape == (30,)
    assert np.all(u >= 0)
    print("PASS test_uniqueness_shape")


# ── Local Density ────────────────────────────────────────────────────────────

def test_density_dense_higher_than_sparse():
    """Points in a tight cluster must have higher density than a lone point."""
    rng = np.random.default_rng(0)
    dense = rng.normal(0, 0.05, size=(50, 2))
    sparse = np.array([[50.0, 50.0]])
    X = np.vstack([dense, sparse])
    rho = compute_density(X, k=5)
    assert rho[:-1].mean() > rho[-1], "dense region should have higher density"
    print(f"PASS test_density_dense_higher_than_sparse (dense={rho[:-1].mean():.3f} > sparse={rho[-1]:.3f})")


def test_density_handles_duplicate_points():
    """Duplicate points give r_k=0; must not produce inf/NaN."""
    X = np.zeros((20, 3))  # all identical
    rho = compute_density(X, k=5)
    assert np.all(np.isfinite(rho)), "duplicates must not produce inf/NaN"
    print("PASS test_density_handles_duplicate_points")


# ── Outlier Score ────────────────────────────────────────────────────────────

def test_outlier_detects_outlier():
    """A clear outlier must score higher than cluster points."""
    rng = np.random.default_rng(0)
    cluster = rng.normal(0, 1, size=(100, 3))
    outlier = np.array([[20.0, 20.0, 20.0]])
    X = np.vstack([cluster, outlier])
    o = compute_outlier_score(X)
    assert o[-1] > np.median(o[:-1]), "outlier should score above cluster median"
    print(f"PASS test_outlier_detects_outlier (outlier o={o[-1]:.3f})")


def test_outlier_range_is_unit_interval():
    X = np.random.default_rng(0).normal(size=(80, 4))
    o = compute_outlier_score(X)
    assert o.min() >= 0.0 and o.max() <= 1.0, "outlier scores must be in [0,1]"
    print("PASS test_outlier_range_is_unit_interval")


# ── Feature Entropy ──────────────────────────────────────────────────────────

def test_entropy_constant_feature_is_zero():
    """A constant feature has zero entropy."""
    X = np.ones((100, 1)) * 3.14
    h = compute_entropy(X)
    assert abs(h) < 1e-9, f"constant feature entropy should be 0, got {h}"
    print("PASS test_entropy_constant_feature_is_zero")


def test_entropy_uniform_higher_than_constant():
    rng = np.random.default_rng(0)
    X_uniform = rng.uniform(0, 1, size=(1000, 1))
    X_constant = np.ones((1000, 1))
    assert compute_entropy(X_uniform) > compute_entropy(X_constant)
    print("PASS test_entropy_uniform_higher_than_constant")


# ── Cluster Separation ───────────────────────────────────────────────────────

def test_cluster_sep_separated_classes_high():
    """Well-separated classes give silhouette near 1."""
    rng = np.random.default_rng(0)
    c0 = rng.normal(0, 0.1, size=(50, 2))
    c1 = rng.normal(10, 0.1, size=(50, 2))
    X = np.vstack([c0, c1])
    y = np.array([0] * 50 + [1] * 50)
    s = compute_cluster_separation(X, y)
    assert s > 0.8, f"separated classes should have high silhouette, got {s:.3f}"
    print(f"PASS test_cluster_sep_separated_classes_high (s={s:.3f})")


def test_cluster_sep_single_class_returns_zero():
    X = np.random.default_rng(0).normal(size=(30, 2))
    y = np.zeros(30, dtype=int)
    assert compute_cluster_separation(X, y) == 0.0
    print("PASS test_cluster_sep_single_class_returns_zero")


# ── Full feature vector + edge cases (likely crash sources) ──────────────────

def test_dpri_features_all_keys_present():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 5))
    y = rng.integers(0, 2, size=100)
    feats = compute_dpri_features(X, y, k=5)
    expected = {
        "uniqueness_mean", "uniqueness_p90", "density_mean", "density_p90",
        "outlier_mean", "outlier_p90", "entropy", "cluster_sep",
    }
    assert set(feats.keys()) == expected, f"missing keys: {expected - set(feats.keys())}"
    print("PASS test_dpri_features_all_keys_present")


def test_dpri_features_no_nan_or_inf():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 5))
    y = rng.integers(0, 2, size=100)
    feats = compute_dpri_features(X, y, k=5)
    for key, val in feats.items():
        assert np.isfinite(val), f"{key} is not finite: {val}"
    print("PASS test_dpri_features_no_nan_or_inf")


def test_dpri_features_high_dimensional():
    """Texas100 has 6169 features — make sure high-dim doesn't break anything."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 6169))
    y = rng.integers(0, 100, size=200)
    feats = compute_dpri_features(X, y, k=5)
    assert all(np.isfinite(v) for v in feats.values())
    print("PASS test_dpri_features_high_dimensional")


def test_dpri_features_many_classes():
    """Purchase100/Texas100 have 100+ classes."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 10))
    y = rng.integers(0, 100, size=300)
    feats = compute_dpri_features(X, y, k=5)
    assert all(np.isfinite(v) for v in feats.values())
    print("PASS test_dpri_features_many_classes")


def test_dpri_features_with_duplicates():
    """Real datasets (esp. binary-feature Purchase100) have many duplicate rows."""
    rng = np.random.default_rng(0)
    base = rng.integers(0, 2, size=(20, 50)).astype(float)
    X = np.repeat(base, 5, axis=0)   # 100 rows, many duplicates
    y = rng.integers(0, 2, size=100)
    feats = compute_dpri_features(X, y, k=5)
    assert all(np.isfinite(v) for v in feats.values())
    print("PASS test_dpri_features_with_duplicates")


if __name__ == "__main__":
    tests = [
        test_uniqueness_isolated_point_is_highest,
        test_uniqueness_shape,
        test_density_dense_higher_than_sparse,
        test_density_handles_duplicate_points,
        test_outlier_detects_outlier,
        test_outlier_range_is_unit_interval,
        test_entropy_constant_feature_is_zero,
        test_entropy_uniform_higher_than_constant,
        test_cluster_sep_separated_classes_high,
        test_cluster_sep_single_class_returns_zero,
        test_dpri_features_all_keys_present,
        test_dpri_features_no_nan_or_inf,
        test_dpri_features_high_dimensional,
        test_dpri_features_many_classes,
        test_dpri_features_with_duplicates,
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
