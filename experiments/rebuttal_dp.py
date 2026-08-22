"""
Rebuttal (#1613), Reviewer B: membership inference under DP-SGD.

For one dataset, trains the MLP family of the main grid (128-64, Adam) with
Opacus DP-SGD at several epsilons plus a non-private baseline with the same
recipe, on the same 50/50 member split as the grid, and attacks every model
with the loss-threshold attack (and optionally LiRA with DP-trained shadows).

    python experiments/rebuttal_dp.py --dataset adult [--epsilons 1,4,8] [--epochs 30] [--lira]
    -> results/rebuttal/dp/adult.json

Requires:  pip install opacus   (not in environment.yml)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.mia.attacks import (  # noqa: E402
    attack_loss_threshold, tpr_at_fpr, accuracy as model_accuracy,
    _predict_proba_correct_class,
)

OUT_DIR = Path("results/rebuttal/dp")


def lira_with_factory(factory, X, y, n_shadow=8, seed=42, eval_n=500):
    """Online LiRA (same logic as src.mia.attacks.attack_lira) with shadows
    built by factory(seed) -> unfitted model; used for DP-trained shadows."""
    from scipy.stats import norm
    from sklearn.metrics import roc_auc_score, roc_curve
    rng = np.random.default_rng(seed)
    n = len(X)
    eval_n = min(eval_n, n)
    eval_idx = rng.choice(n, eval_n, replace=False)
    masks = np.zeros((n_shadow, n), dtype=bool)
    confs = np.zeros((n_shadow, n), dtype=np.float64)
    for i in range(n_shadow):
        train_mask = rng.random(n) < 0.5
        masks[i] = train_mask
        shadow = factory(seed + i)
        shadow.fit(X[train_mask], y[train_mask])
        confs[i] = _predict_proba_correct_class(shadow, X, y)
    eps = 1e-6
    cc = np.clip(confs, eps, 1 - eps)
    logit = np.log(cc / (1 - cc))
    scores, labels = [], []
    for j in eval_idx:
        col_mask, col_logit = masks[:, j], logit[:, j]
        for t in range(n_shadow):
            sel = np.ones(n_shadow, dtype=bool); sel[t] = False
            in_refs, out_refs = col_logit[sel & col_mask], col_logit[sel & ~col_mask]
            if len(in_refs) < 2 or len(out_refs) < 2:
                continue
            s = (norm.logpdf(col_logit[t], in_refs.mean(), in_refs.std() + 1e-6)
                 - norm.logpdf(col_logit[t], out_refs.mean(), out_refs.std() + 1e-6))
            scores.append(s); labels.append(int(col_mask[t]))
    if len(set(labels)) < 2:
        return np.array([0, 1]), np.array([0, 1]), 0.5
    scores, labels = np.array(scores), np.array(labels)
    fpr, tpr, _ = roc_curve(labels, scores)
    return fpr, tpr, float(roc_auc_score(labels, scores))


class DPMLP:
    """sklearn-like MLP (128-64, ReLU) trained with Opacus DP-SGD when
    epsilon is not None, otherwise with plain Adam and the same recipe."""

    def __init__(self, epsilon=None, hidden=(128, 64), epochs=30, lr=1e-3,
                 batch_size=256, max_grad_norm=1.0, delta=1e-5, seed=42):
        self.epsilon = epsilon
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.delta = delta
        self.seed = seed
        self.net = None
        self.classes_ = None
        self.eps_spent = None
        self.device = None

    def _build(self, in_dim, n_classes):
        import torch.nn as nn
        layers, prev = [], in_dim
        for h in self.hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        return nn.Sequential(*layers)

    def fit(self, X, y):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.classes_ = np.unique(y)
        lut = {c: i for i, c in enumerate(self.classes_)}
        y_int = np.array([lut[c] for c in y], dtype=np.int64)

        net = self._build(X.shape[1], len(self.classes_)).to(self.device)
        Xt = torch.from_numpy(X.astype(np.float32)).to(self.device)
        yt = torch.from_numpy(y_int).to(self.device)
        loader = DataLoader(TensorDataset(Xt, yt), batch_size=self.batch_size, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        engine = None

        if self.epsilon is not None:
            from opacus import PrivacyEngine
            from opacus.validators import ModuleValidator
            net = ModuleValidator.fix(net)
            optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
            engine = PrivacyEngine()
            net, optimizer, loader = engine.make_private_with_epsilon(
                module=net, optimizer=optimizer, data_loader=loader,
                epochs=self.epochs, target_epsilon=float(self.epsilon),
                target_delta=self.delta, max_grad_norm=self.max_grad_norm,
            )
        else:
            optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)

        net.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                if len(yb) == 0:          # Poisson sampling can yield empty batches
                    continue
                optimizer.zero_grad()
                criterion(net(xb), yb).backward()
                optimizer.step()

        self.net = net
        self.eps_spent = float(engine.get_epsilon(self.delta)) if engine is not None else None
        return self

    def predict_proba(self, X, batch_size=4096):
        import torch
        self.net.eval()
        out = []
        with torch.no_grad():
            for s in range(0, len(X), batch_size):
                xb = torch.from_numpy(X[s:s + batch_size].astype(np.float32)).to(self.device)
                out.append(torch.softmax(self.net(xb), dim=1).cpu().numpy())
        return np.concatenate(out, axis=0)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def run(dataset, data_dir, epsilons, epochs, seed, do_lira, lira_shadows, lira_eval_n):
    path = Path(data_dir) / f"{dataset}.npz"
    if not path.exists():
        print(f"SKIP: {path} not found")
        return None
    d = np.load(path)
    X, y = d["X"], d["y"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=seed, stratify=y)

    results = {}
    for eps in [None] + list(epsilons):
        label = "inf" if eps is None else str(eps)
        t0 = time.time()
        model = DPMLP(epsilon=eps, epochs=epochs, seed=seed).fit(X_train, y_train)
        fpr, tpr, auc = attack_loss_threshold(model, X_train, y_train, X_test, y_test)
        rec = {
            "epsilon_target": eps,
            "epsilon_spent": model.eps_spent,
            "loss_auc": float(auc),
            "loss_tpr_at_fpr_01": tpr_at_fpr(fpr, tpr, 0.01),
            "loss_tpr_at_fpr_001": tpr_at_fpr(fpr, tpr, 0.001),
            "train_acc": model_accuracy(model, X_train, y_train),
            "test_acc": model_accuracy(model, X_test, y_test),
            "elapsed_sec": round(time.time() - t0, 1),
        }
        rec["gen_gap"] = rec["train_acc"] - rec["test_acc"]
        if do_lira:
            t1 = time.time()
            fpr_l, tpr_l, auc_l = lira_with_factory(
                lambda s, e=eps: DPMLP(epsilon=e, epochs=epochs, seed=s),
                X, y, n_shadow=lira_shadows, seed=seed, eval_n=lira_eval_n)
            rec["lira_auc"] = float(auc_l)
            rec["lira_tpr_at_fpr_01"] = tpr_at_fpr(fpr_l, tpr_l, 0.01)
            rec["lira_elapsed_sec"] = round(time.time() - t1, 1)
        print(f"  eps={label:>4}: loss-AUC={auc:.3f}  test_acc={rec['test_acc']:.3f}  "
              f"spent={rec['epsilon_spent']}" + (f"  LiRA={rec['lira_auc']:.3f}" if do_lira else ""))
        results[label] = rec

    return {
        "dataset": dataset,
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "n_classes": int(len(np.unique(y))),
        "epochs": epochs,
        "seed": seed,
        "results": results,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--out_dir", default=str(OUT_DIR))
    ap.add_argument("--epsilons", default="1,4,8")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lira", action="store_true", help="also run LiRA with DP-trained shadows")
    ap.add_argument("--lira_shadows", type=int, default=8)
    ap.add_argument("--lira_eval_n", type=int, default=500)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.dataset}.json"
    if out_path.exists():
        print(f"Already done: {out_path}")
        return
    eps = [float(e) for e in args.epsilons.split(",") if e.strip()]
    print(f"DP-SGD run: {args.dataset}  epsilons={eps}  epochs={args.epochs}  lira={args.lira}")
    res = run(args.dataset, args.data_dir, eps, args.epochs, args.seed,
              args.lira, args.lira_shadows, args.lira_eval_n)
    if res is None:
        return
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
