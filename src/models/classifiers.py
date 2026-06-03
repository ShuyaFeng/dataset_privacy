"""
Three model families used in the MIA grid experiment.
All expose a sklearn-compatible fit/predict_proba interface.

MLP  → PyTorch, auto-uses GPU when available (CUDA) else CPU.
XGBoost → uses GPU tree method when CUDA is available.
RF   → sklearn, CPU only (fast enough for tabular data).
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def _cuda_available() -> bool:
    return torch.cuda.is_available()


# ── PyTorch MLP with sklearn-compatible interface ────────────────────────────

class TorchMLP:
    """
    Sklearn-compatible MLP backed by PyTorch.
    Automatically uses CUDA if available; falls back to CPU.
    Exposes: fit, predict_proba, predict, classes_
    """

    def __init__(
        self,
        hidden_sizes: tuple = (128, 64),
        epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 256,
        seed: int = 42,
    ):
        self.hidden_sizes = hidden_sizes
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed
        self.device = torch.device("cuda" if _cuda_available() else "cpu")
        self.net = None
        self.classes_ = None
        self._label_map = None

    def _build_net(self, in_dim: int, n_classes: int) -> nn.Module:
        layers = []
        prev = in_dim
        for h in self.hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        return nn.Sequential(*layers)

    def fit(self, X: np.ndarray, y: np.ndarray):
        torch.manual_seed(self.seed)
        self.classes_ = np.unique(y)
        self._label_map = {c: i for i, c in enumerate(self.classes_)}
        y_int = np.array([self._label_map[c] for c in y], dtype=np.int64)

        self.net = self._build_net(X.shape[1], len(self.classes_)).to(self.device)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        Xt = torch.from_numpy(X.astype(np.float32)).to(self.device)
        yt = torch.from_numpy(y_int).to(self.device)
        loader = DataLoader(
            TensorDataset(Xt, yt),
            batch_size=self.batch_size,
            shuffle=True,
        )

        self.net.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                criterion(self.net(xb), yb).backward()
                optimizer.step()
        return self

    def predict_proba(self, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        self.net.eval()
        results = []
        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                xb = torch.from_numpy(
                    X[start : start + batch_size].astype(np.float32)
                ).to(self.device)
                results.append(torch.softmax(self.net(xb), dim=1).cpu().numpy())
        return np.concatenate(results, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# ── Factory ──────────────────────────────────────────────────────────────────

def get_model(name: str, n_classes: int, seed: int = 42):
    """Return an untrained model by name."""
    if name == "mlp":
        return TorchMLP(
            hidden_sizes=(128, 64),
            epochs=100,
            lr=1e-3,
            batch_size=256,
            seed=seed,
        )
    if name == "xgboost":
        # use GPU tree method when CUDA is available
        device = "cuda" if _cuda_available() else "cpu"
        return XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            eval_metric="logloss",
            random_state=seed,
            verbosity=0,
            n_jobs=4,
            device=device,
        )
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=1,
            n_jobs=4,
            random_state=seed,
        )
    raise ValueError(f"Unknown model: {name}")


MODEL_NAMES = ["mlp", "xgboost", "rf"]
