"""
splitting.py — Train / validation / test split utilities (student-implementable).

``split_data`` implements the **5-fold stratified outer CV** + **inner stratified
validation** slice used for ``tests/TESTS_Ratio.md`` (~73.77% mean test AUROC
over seeds 42–44).

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
    n_splits: int = 5,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    """Split dataset indices into train, validation, and test subsets.

    The default strategy performs stratified K-fold evaluation. For each outer
    fold, one stratified fold is held out as ``idx_test`` and a stratified
    validation slice is carved from the remaining data for threshold tuning.

    Args:
        y:            Label array of shape ``(N,)`` with values in ``{0, 1}``.
                      Used for stratification.
        df:           Optional full DataFrame (same row order as ``y``).
                      Required for group-aware splits.
        test_size:    Kept for API compatibility; the held-out test fraction is
                      controlled by ``n_splits``.
        val_size:     Approximate fraction of all samples reserved for
                      validation inside each fold.
        random_state: Random seed for reproducible splits.
        n_splits:     Number of stratified outer folds.

    Returns:
        A list of ``(idx_train, idx_val, idx_test)`` tuples of integer index
        arrays.  ``idx_val`` may be ``None``.

    Student task:
        Replace or extend the skeleton below.  The only contract is that the
        function returns the list described above.
    """

    del df, test_size  # accepted by the public API but unused in this strategy

    idx = np.arange(len(y))
    y = np.asarray(y)

    outer = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    splits: list[tuple[np.ndarray, np.ndarray | None, np.ndarray]] = []
    for fold_id, (idx_train_val, idx_test) in enumerate(outer.split(idx, y)):
        # Keep ``val_size`` as an approximate fraction of the full dataset.
        # Since ``idx_train_val`` excludes the current test fold, convert that
        # requested full-data fraction to a fraction of the train+val pool.
        relative_val = val_size / (len(idx_train_val) / len(idx))
        relative_val = min(max(relative_val, 0.0), 0.5)

        if relative_val == 0.0:
            idx_train = idx_train_val
            idx_val = None
        else:
            idx_train, idx_val = train_test_split(
                idx_train_val,
                test_size=relative_val,
                random_state=random_state + fold_id,
                stratify=y[idx_train_val],
            )

        splits.append(
            (
                np.asarray(idx_train, dtype=int),
                None if idx_val is None else np.asarray(idx_val, dtype=int),
                np.asarray(idx_test, dtype=int),
            )
        )

    return splits

