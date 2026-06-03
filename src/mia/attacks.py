"""
Three MIA attacks used in the ground-truth grid.

All return (fpr_array, tpr_array, auc_score) for each member/non-member split.

Attacks implemented:
  1. loss_threshold  — Yeom et al. 2018, baseline
  2. shadow_model    — Shokri et al. 2017, simplified (2 shadow models)
  3. lira            — Carlini et al. 2022, simplified (16 shadow models)
                       Full LiRA uses 64+ shadows; 16 is a fast approximation.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from src.models.classifiers import get_model


# ── helpers ─────────────────────────────────────────────────────────────────

def _split_members_nonmembers(X, y, train_idx, test_idx):
    """Return (X_mem, y_mem, X_non, y_non)."""
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def _predict_proba_correct_class(model, X, y):
    """Probability assigned to the true class label for each sample."""
    proba = model.predict_proba(X)          # (n, k)
    classes = np.array(model.classes_)
    # map y values to column indices — vectorized, no Python loop
    col_idx = np.searchsorted(classes, y)   # works because classes_ is sorted
    return proba[np.arange(len(y)), col_idx]


# ── Attack 1: Loss Threshold ─────────────────────────────────────────────────

def attack_loss_threshold(model, X_train, y_train, X_test, y_test):
    """
    Members have higher confidence on their true label.
    Score = confidence of correct class. Higher → more likely member.
    """
    scores_mem = _predict_proba_correct_class(model, X_train, y_train)
    scores_non = _predict_proba_correct_class(model, X_test, y_test)

    labels = np.concatenate([np.ones(len(scores_mem)), np.zeros(len(scores_non))])
    scores = np.concatenate([scores_mem, scores_non])
    auc = roc_auc_score(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    return fpr, tpr, float(auc)


# ── Attack 2: Shadow Model ───────────────────────────────────────────────────

def attack_shadow_model(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    n_shadow: int = 4,
    seed: int = 42,
):
    """
    Train n_shadow models on disjoint subsets; collect (confidence, in/out) pairs;
    train a meta-classifier; evaluate on a held-out split.

    Returns (fpr, tpr, auc) on the held-out evaluation set.
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    half = n // 2

    meta_X, meta_y = [], []

    for i in range(n_shadow):
        idx = rng.permutation(n)
        train_idx, test_idx = idx[:half], idx[half:]

        shadow = get_model(model_name, n_classes=len(np.unique(y)), seed=seed + i)
        shadow.fit(X[train_idx], y[train_idx])

        conf_in  = _predict_proba_correct_class(shadow, X[train_idx], y[train_idx])
        conf_out = _predict_proba_correct_class(shadow, X[test_idx],  y[test_idx])

        meta_X.extend(conf_in.reshape(-1, 1))
        meta_X.extend(conf_out.reshape(-1, 1))
        meta_y.extend([1] * len(conf_in))
        meta_y.extend([0] * len(conf_out))

    meta_X = np.array(meta_X)
    meta_y = np.array(meta_y)

    # evaluate meta-classifier via cross-validation
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    aucs = []
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
    for tr, te in skf.split(meta_X, meta_y):
        clf = LogisticRegression()
        clf.fit(meta_X[tr], meta_y[tr])
        preds = clf.predict_proba(meta_X[te])[:, 1]
        aucs.append(roc_auc_score(meta_y[te], preds))

    # return full curve on last fold
    clf = LogisticRegression()
    clf.fit(meta_X[tr], meta_y[tr])
    preds = clf.predict_proba(meta_X[te])[:, 1]
    fpr, tpr, _ = roc_curve(meta_y[te], preds)
    auc = float(np.mean(aucs))
    return fpr, tpr, auc


# ── Attack 3: LiRA (simplified, 16 shadow models) ────────────────────────────

def attack_lira(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    n_shadow: int = 16,
    seed: int = 42,
):
    """
    LiRA: Carlini et al. 2022.
    For each target sample, compare its loss under models trained WITH vs WITHOUT it.
    Simplified version uses n_shadow=16 (full paper uses 64+).

    Returns (fpr, tpr, auc).
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    n_classes = len(np.unique(y))

    # For efficiency: evaluate on a random subset of 500 target samples
    eval_n = min(500, n)
    eval_idx = rng.choice(n, eval_n, replace=False)

    # For each target sample, track: list of (in_loss, out_loss)
    in_scores  = [[] for _ in range(n)]
    out_scores = [[] for _ in range(n)]

    for i in range(n_shadow):
        # each shadow model is trained on a random 50% subset
        train_mask = rng.random(n) < 0.5
        train_idx  = np.where(train_mask)[0]

        shadow = get_model(model_name, n_classes=n_classes, seed=seed + i)
        shadow.fit(X[train_idx], y[train_idx])

        # confidence on correct class = proxy for -loss
        conf_all = _predict_proba_correct_class(shadow, X, y)

        for j in eval_idx:
            if train_mask[j]:
                in_scores[j].append(conf_all[j])
            else:
                out_scores[j].append(conf_all[j])

    # Compute LiRA score for each evaluated sample
    lira_scores = []
    true_labels = []  # 1 = member (we define membership as being in the first half)

    half = n // 2
    for j in eval_idx:
        ins  = np.array(in_scores[j])
        outs = np.array(out_scores[j])
        if len(ins) < 2 or len(outs) < 2:
            continue
        # likelihood ratio: mean in_conf - mean out_conf
        score = float(np.mean(ins) - np.mean(outs))
        lira_scores.append(score)
        true_labels.append(1 if j < half else 0)

    if len(set(true_labels)) < 2:
        # degenerate case: return 0.5
        return np.array([0, 1]), np.array([0, 1]), 0.5

    lira_scores = np.array(lira_scores)
    true_labels = np.array(true_labels)
    auc = roc_auc_score(true_labels, lira_scores)
    fpr, tpr, _ = roc_curve(true_labels, lira_scores)
    return fpr, tpr, float(auc)


ATTACK_NAMES = ["loss_threshold", "shadow_model", "lira"]
