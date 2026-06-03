"""
Three model families used in the MIA grid experiment.
All expose a sklearn-compatible fit/predict_proba interface.
"""

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def get_model(name: str, n_classes: int, seed: int = 42):
    """Return an untrained model by name."""
    if name == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=200,
            random_state=seed,
            early_stopping=True,
            validation_fraction=0.1,
        )
    if name == "xgboost":
        return XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=seed,
            verbosity=0,
            n_jobs=4,
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
