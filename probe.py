"""
probe.py — Hallucination probe: CatBoost direct (no PCA).

CatBoost handles feature selection internally via ordered boosting.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


class HallucinationProbe(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self._dummy = nn.Parameter(torch.zeros(1))
        self._scaler = StandardScaler()
        self._model: CatBoostClassifier | None = None
        self._threshold: float = 0.5

    def _make_model(self) -> CatBoostClassifier:
        return CatBoostClassifier(
            iterations=500, depth=4, learning_rate=0.03,
            l2_leaf_reg=10, auto_class_weights="Balanced",
            rsm=0.3, random_seed=42, verbose=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("Use predict/predict_proba instead of forward()")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        X_scaled = self._scaler.fit_transform(X)

        self._model = self._make_model()
        self._model.fit(X_scaled, y)

        self._tune_threshold_internal(X_scaled, y)

        return self

    def _tune_threshold_internal(self, X: np.ndarray, y: np.ndarray) -> None:
        if len(np.unique(y)) < 2:
            return

        all_probs = np.zeros(len(y))
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        for tr_idx, va_idx in skf.split(X, y):
            fold_model = self._make_model()
            fold_model.fit(X[tr_idx], y[tr_idx])
            all_probs[va_idx] = fold_model.predict_proba(X[va_idx])[:, 1]

        best_threshold, best_f1 = 0.5, -1.0
        for t in np.linspace(0.2, 0.8, 61):
            score = f1_score(y, (all_probs >= t).astype(int), zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_threshold = float(t)
        self._threshold = best_threshold

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray,
    ) -> "HallucinationProbe":
        probs = self.predict_proba(X_val)[:, 1]
        candidates = np.unique(np.concatenate([probs, np.linspace(0.0, 1.0, 101)]))

        best_threshold, best_f1 = 0.5, -1.0
        for t in candidates:
            score = f1_score(y_val, (probs >= t).astype(int), zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_threshold = float(t)
        self._threshold = best_threshold
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self._scaler.transform(X)
        return self._model.predict_proba(X_scaled)
