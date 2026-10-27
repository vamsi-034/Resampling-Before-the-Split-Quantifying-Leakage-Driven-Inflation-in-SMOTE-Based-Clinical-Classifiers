"""
The two pipelines under comparison.

leaky_pipeline    : SMOTE applied to the FULL dataset, THEN split.
                    Synthetic minority points can be interpolated from
                    neighbors that end up on opposite sides of the
                    train/test boundary, and duplicated information
                    can appear in both sets. This is the bug.

correct_pipeline  : split FIRST, THEN SMOTE applied to the training
                    fold only. The test fold is never touched by the
                    resampler and keeps the original class balance.

Both pipelines return (X_train, X_test, y_train, y_test) so a caller
can plug either output into an identical downstream classifier and
scaler, isolating the split-order variable.
"""

from dataclasses import dataclass

import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split


@dataclass
class SplitResult:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    pipeline: str  # "leaky" or "correct"


def leaky_pipeline(X, y, seed, k_neighbors=5, test_size=0.2) -> SplitResult:
    smote = SMOTE(random_state=seed, k_neighbors=k_neighbors)
    X_res, y_res = smote.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=test_size, random_state=seed, stratify=y_res
    )
    return SplitResult(X_train, X_test, y_train, y_test, pipeline="leaky")


def correct_pipeline(X, y, seed, k_neighbors=5, test_size=0.2) -> SplitResult:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    smote = SMOTE(random_state=seed, k_neighbors=k_neighbors)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    return SplitResult(X_train_res, X_test, y_train_res, y_test, pipeline="correct")


PIPELINES = {
    "leaky": leaky_pipeline,
    "correct": correct_pipeline,
}
