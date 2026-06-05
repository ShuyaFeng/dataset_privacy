"""
Unit tests for the MIA attacks — run LOCALLY before submitting cluster jobs.

The key test is `test_lira_detects_leakage`: in an overfit (leaky) scenario,
a CORRECT attack must score AUC well above 0.5. The old buggy LiRA used an
arbitrary `j < half` membership label and produced AUC ~0.5 even when the
model leaked badly — this test would have caught that.

Run:
    python -m pytest tests/test_attacks.py -v
    # or without pytest:
    python tests/test_attacks.py
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mia.attacks import (
    attack_loss_threshold,
    attack_shadow_model,
    attack_lira,
    _predict_proba_correct_class,
)
from src.models.classifiers import get_model


# ── helpers ──────────────────────────────────────────────────────────────────

def _leaky_data(seed=0, n=400):
    """A dataset with real signal; an unregularized RF will overfit and leak."""
    X, y = make_classification(
        n_samples=n, n_features=20, n_informative=10, n_redundant=5,
        n_classes=2, random_state=seed,
    )
    return X.astype(np.float32), y.astype(np.int64)


# ── Test 1: vectorized confidence extraction ─────────────────────────────────

def test_predict_proba_correct_class():
    """The np.searchsorted vectorization must match manual indexing."""
    class MockModel:
        classes_ = np.array([0, 1, 2])
        def predict_proba(self, X):
            return np.array([
                [0.1, 0.7, 0.2],   # sample 0
                [0.6, 0.3, 0.1],   # sample 1
                [0.2, 0.2, 0.6],   # sample 2
            ])
    m = MockModel()
    y = np.array([1, 0, 2])
    out = _predict_proba_correct_class(m, np.zeros((3, 3)), y)
    expected = np.array([0.7, 0.6, 0.6])   # proba[i, y[i]]
    assert np.allclose(out, expected), f"got {out}, expected {expected}"
    print("PASS test_predict_proba_correct_class")


def test_predict_proba_non_contiguous_classes():
    """searchsorted must handle non-contiguous class labels (e.g. [0, 5, 9])."""
    class MockModel:
        classes_ = np.array([0, 5, 9])
        def predict_proba(self, X):
            return np.array([
                [0.1, 0.8, 0.1],   # column 1 == class 5
                [0.7, 0.2, 0.1],   # column 0 == class 0
            ])
    m = MockModel()
    y = np.array([5, 0])
    out = _predict_proba_correct_class(m, np.zeros((2, 3)), y)
    expected = np.array([0.8, 0.7])
    assert np.allclose(out, expected), f"got {out}, expected {expected}"
    print("PASS test_predict_proba_non_contiguous_classes")


# ── Test 2: all three attacks detect leakage ─────────────────────────────────

def test_loss_threshold_detects_leakage():
    X, y = _leaky_data()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, random_state=0)
    model = get_model("rf", n_classes=2, seed=0)
    model.fit(Xtr, ytr)
    _, _, auc = attack_loss_threshold(model, Xtr, ytr, Xte, yte)
    assert auc > 0.55, f"loss_threshold AUC={auc:.4f} — should detect leakage"
    assert auc <= 1.0
    print(f"PASS test_loss_threshold_detects_leakage (AUC={auc:.4f})")


def test_shadow_model_detects_leakage():
    X, y = _leaky_data()
    _, _, auc = attack_shadow_model("rf", X, y, n_shadow=4, seed=0)
    assert auc > 0.55, f"shadow_model AUC={auc:.4f} — should detect leakage"
    assert auc <= 1.0
    print(f"PASS test_shadow_model_detects_leakage (AUC={auc:.4f})")


def test_lira_detects_leakage():
    """THE KEY TEST: this fails on the old buggy LiRA (which gave ~0.5)."""
    X, y = _leaky_data()
    _, _, auc = attack_lira("rf", X, y, n_shadow=8, seed=0)
    assert auc > 0.55, f"LiRA AUC={auc:.4f} — should detect leakage (old bug gave ~0.5)"
    assert auc <= 1.0
    print(f"PASS test_lira_detects_leakage (AUC={auc:.4f})")


# ── Test 3: attacks never score systematically below random ──────────────────

def test_lira_not_below_random():
    """A correct attack must never be systematically below 0.5."""
    for seed in [0, 1, 2]:
        X, y = _leaky_data(seed=seed)
        _, _, auc = attack_lira("rf", X, y, n_shadow=8, seed=seed)
        assert auc >= 0.45, f"LiRA AUC={auc:.4f} at seed={seed} — below random!"
    print("PASS test_lira_not_below_random")


def test_lira_auc_in_valid_range():
    """AUC must always be a valid [0,1] float."""
    X, y = _leaky_data()
    _, _, auc = attack_lira("rf", X, y, n_shadow=8, seed=0)
    assert 0.0 <= auc <= 1.0
    assert isinstance(auc, float)
    print(f"PASS test_lira_auc_in_valid_range (AUC={auc:.4f})")


# ── manual runner (no pytest needed) ─────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_predict_proba_correct_class,
        test_predict_proba_non_contiguous_classes,
        test_loss_threshold_detects_leakage,
        test_shadow_model_detects_leakage,
        test_lira_detects_leakage,
        test_lira_not_below_random,
        test_lira_auc_in_valid_range,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{'='*50}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
