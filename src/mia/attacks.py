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

def attack_loss_threshold(model, X_train, y_train, X_test, y_test,
                          return_details: bool = False):
    """
    Members have higher confidence on their true label.
    Score = confidence of correct class. Higher → more likely member.

    With return_details=True a 4th value is returned: a dict with the
    model's train/test accuracy (generalization gap), used by the rebuttal
    failure analysis. Default behaviour is unchanged.
    """
    scores_mem = _predict_proba_correct_class(model, X_train, y_train)
    scores_non = _predict_proba_correct_class(model, X_test, y_test)

    labels = np.concatenate([np.ones(len(scores_mem)), np.zeros(len(scores_non))])
    scores = np.concatenate([scores_mem, scores_non])
    auc = roc_auc_score(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    if return_details:
        details = {
            "train_acc": float(accuracy(model, X_train, y_train)),
            "test_acc": float(accuracy(model, X_test, y_test)),
            "n_scores": int(len(scores)),
        }
        return fpr, tpr, float(auc), details
    return fpr, tpr, float(auc)


def accuracy(model, X, y):
    """Classification accuracy via predict_proba (works for all model families)."""
    proba = model.predict_proba(X)
    classes = np.array(model.classes_)
    return float((classes[np.argmax(proba, axis=1)] == y).mean())


# ── Attack 2: Shadow Model ───────────────────────────────────────────────────

def attack_shadow_model(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    n_shadow: int = 4,
    seed: int = 42,
    return_details: bool = False,
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
    tr_accs, te_accs = [], []

    for i in range(n_shadow):
        idx = rng.permutation(n)
        train_idx, test_idx = idx[:half], idx[half:]

        shadow = get_model(model_name, n_classes=len(np.unique(y)), seed=seed + i)
        shadow.fit(X[train_idx], y[train_idx])
        tr_accs.append(accuracy(shadow, X[train_idx], y[train_idx]))
        te_accs.append(accuracy(shadow, X[test_idx], y[test_idx]))

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
    if return_details:
        details = {"train_acc": float(np.mean(tr_accs)),
                   "test_acc": float(np.mean(te_accs)),
                   "n_scores": int(len(te))}
        return fpr, tpr, auc, details
    return fpr, tpr, auc


# ── Attack 3: LiRA (online, 16 shadow models) ────────────────────────────────

def attack_lira(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    n_shadow: int = 16,
    seed: int = 42,
    eval_n: int = 500,
    return_details: bool = False,
):
    """
    Online LiRA: Carlini et al. 2022.

    Train n_shadow models, each on a random 50% subset. For each
    (target model t, eval sample j) pair, use the OTHER shadow models as
    references to estimate j's IN and OUT confidence distributions, then
    score target model t's confidence on j via a Gaussian likelihood ratio.

    The membership label is the GROUND TRUTH: whether j was actually in the
    training set of target model t. This is the correct evaluation — the
    previous version used an arbitrary `j < half` label, which is unrelated
    to actual membership and produced AUC ~0.5.

    Returns (fpr, tpr, auc).
    """
    from scipy.stats import norm

    rng = np.random.default_rng(seed)
    n = len(X)
    n_classes = len(np.unique(y))

    # Evaluate on a random subset of target samples for efficiency.
    # eval_n=500 reproduces the submitted grid; the rebuttal grid uses a
    # larger eval_n so that TPR at 0.1% FPR rests on more than a handful of
    # non-member scores.
    # The first min(500, n) targets are drawn exactly as in the submitted
    # grid, so the rng state (and hence every shadow model) is bit-identical
    # to the submission for any eval_n. Extra targets come from a separate
    # stream and are disjoint from the first batch.
    base_n = min(500, n)
    eval_idx = rng.choice(n, base_n, replace=False)
    eval_n = min(eval_n, n)
    if eval_n > base_n:
        rng_extra = np.random.default_rng(seed + 7919)
        rest = np.setdiff1d(np.arange(n), eval_idx)
        extra = rng_extra.choice(rest, eval_n - base_n, replace=False)
        eval_idx = np.concatenate([eval_idx, extra])

    # Train all shadow models; record membership masks and per-sample confidence.
    masks = np.zeros((n_shadow, n), dtype=bool)     # masks[i, j] = j in shadow i train set
    confs = np.zeros((n_shadow, n), dtype=np.float64)
    tr_accs, te_accs = [], []

    for i in range(n_shadow):
        train_mask = rng.random(n) < 0.5
        masks[i] = train_mask
        shadow = get_model(model_name, n_classes=n_classes, seed=seed + i)
        shadow.fit(X[train_mask], y[train_mask])
        proba = shadow.predict_proba(X)
        classes = np.array(shadow.classes_)
        confs[i] = proba[np.arange(n), np.searchsorted(classes, y)]
        pred_ok = classes[np.argmax(proba, axis=1)] == y
        tr_accs.append(float(pred_ok[train_mask].mean()))
        te_accs.append(float(pred_ok[~train_mask].mean()))

    # LiRA operates in logit space.
    eps = 1e-6
    cc = np.clip(confs, eps, 1 - eps)
    logit_confs = np.log(cc / (1 - cc))

    scores, labels = [], []

    for j in eval_idx:
        col_mask  = masks[:, j]          # (n_shadow,) which shadows had j IN
        col_logit = logit_confs[:, j]    # (n_shadow,) logit-confidence on j

        for t in range(n_shadow):
            # leave target t out; use the rest as references
            sel = np.ones(n_shadow, dtype=bool)
            sel[t] = False
            in_refs  = col_logit[sel &  col_mask]
            out_refs = col_logit[sel & ~col_mask]
            if len(in_refs) < 2 or len(out_refs) < 2:
                continue

            mu_in,  std_in  = in_refs.mean(),  in_refs.std()  + 1e-6
            mu_out, std_out = out_refs.mean(), out_refs.std() + 1e-6

            # log-likelihood ratio: IN vs OUT for target model's confidence
            s = (norm.logpdf(col_logit[t], mu_in,  std_in)
                 - norm.logpdf(col_logit[t], mu_out, std_out))
            scores.append(s)
            labels.append(int(col_mask[t]))   # ground-truth membership

    if len(set(labels)) < 2:
        if return_details:
            return np.array([0, 1]), np.array([0, 1]), 0.5, {"degenerate": True}
        return np.array([0, 1]), np.array([0, 1]), 0.5

    scores = np.array(scores)
    labels = np.array(labels)
    auc = roc_auc_score(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    if return_details:
        details = {"train_acc": float(np.mean(tr_accs)),
                   "test_acc": float(np.mean(te_accs)),
                   "n_scores": int(len(scores)),
                   "n_nonmember_scores": int((labels == 0).sum()),
                   "eval_n": int(eval_n)}
        return fpr, tpr, float(auc), details
    return fpr, tpr, float(auc)


def tpr_at_fpr(fpr, tpr, target_fpr: float) -> float:
    """Largest TPR achievable at FPR <= target_fpr on a ROC curve (the
    convention of Carlini et al. 2022; no interpolation above the target)."""
    fpr = np.asarray(fpr)
    tpr = np.asarray(tpr)
    ok = fpr <= target_fpr
    return float(tpr[ok].max()) if ok.any() else 0.0


ATTACK_NAMES = ["loss_threshold", "shadow_model", "lira"]
