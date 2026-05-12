"""
splitting.py — Train / validation / test split utilities (student-implementable).

``split_data`` receives the label array ``y`` and, optionally, the full
DataFrame ``df`` (for group-aware splits).  It must return a list of
``(idx_train, idx_val, idx_test)`` tuples of integer index arrays.

Contract
--------
* ``idx_train``, ``idx_val``, ``idx_test`` are 1-D NumPy arrays of integer
  indices into the full dataset.
* ``idx_val`` may be ``None`` if no separate validation fold is needed.
* All indices must be non-overlapping; together they must cover every sample.
* Return a **list** — one element for a single split, K elements for k-fold.
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
    """Split dataset indices into stratified train, validation, and test folds.

    The strategy uses an outer stratified k-fold split for a more stable
    estimate on the small labelled dataset, then carves a stratified validation
    split out of each fold's non-test samples for threshold tuning.

    Args:
        y:            Label array of shape ``(N,)`` with values in ``{0, 1}``.
                      Used for stratification.
        df:           Optional full DataFrame (same row order as ``y``).
                      Required for group-aware splits.
        test_size:    Fraction of samples reserved for the held-out test set.
        val_size:     Fraction of samples reserved for validation.
        random_state: Random seed for reproducible splits.

    Returns:
        A list of ``(idx_train, idx_val, idx_test)`` tuples of integer index
        arrays.  ``idx_val`` may be ``None``.

    Student task:
        Replace or extend the skeleton below.  The only contract is that the
        function returns the list described above.
    """

    idx = np.arange(len(y))
    y = np.asarray(y).astype(int)

    n_splits = 5
    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    splits: list[tuple[np.ndarray, np.ndarray | None, np.ndarray]] = []
    for fold_idx, (idx_train_val, idx_test) in enumerate(splitter.split(idx, y)):
        idx_train_val = idx_train_val.astype(int)
        idx_test = idx_test.astype(int)

        if val_size <= 0.0:
            idx_train = idx_train_val
            idx_val = None
        else:
            relative_val = min(0.30, max(0.10, val_size / (1.0 - 1.0 / n_splits)))
            idx_train, idx_val = train_test_split(
                idx_train_val,
                test_size=relative_val,
                random_state=random_state + fold_idx,
                stratify=y[idx_train_val],
            )
            idx_train = idx_train.astype(int)
            idx_val = idx_val.astype(int)

        splits.append((np.sort(idx_train), None if idx_val is None else np.sort(idx_val), np.sort(idx_test)))

    return splits
