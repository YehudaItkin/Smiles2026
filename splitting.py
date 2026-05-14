"""
splitting.py — Train / validation / test split utilities (student-implementable).

5-fold Stratified Cross-Validation with per-fold validation split for
threshold tuning.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    splits = []

    for train_val_idx, test_idx in skf.split(np.arange(len(y)), y):
        relative_val = val_size / (1.0 - 1.0 / 5.0)
        idx_train, idx_val = train_test_split(
            train_val_idx,
            test_size=relative_val,
            random_state=random_state,
            stratify=y[train_val_idx],
        )
        splits.append((idx_train, idx_val, test_idx))

    return splits
