"""
Generalizes pipelines.py's leaky/correct pipeline pair across resampler
choice, to test whether the leakage effect quantified for SMOTE also
appears for related over-sampling variants (ADASYN, Borderline-SMOTE).

Same core manipulation as pipelines.py: leaky = resample-then-split,
correct = split-then-resample-on-training-fold-only. Only the resampler
class differs.
"""

from dataclasses import dataclass

import numpy as np
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from sklearn.model_selection import train_test_split


RESAMPLERS = {
    "smote": SMOTE,
    "adasyn": ADASYN,
    "borderline_smote": BorderlineSMOTE,
}


@dataclass
class SplitResult:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    pipeline: str
    resampler: str


def _make_resampler(name, seed, k_neighbors):
    cls = RESAMPLERS[name]
    if name == "adasyn":
        return cls(random_state=seed, n_neighbors=k_neighbors)
    return cls(random_state=seed, k_neighbors=k_neighbors)


def leaky_pipeline_resampler(X, y, seed, resampler_name="smote", k_neighbors=5, test_size=0.2):
    resampler = _make_resampler(resampler_name, seed, k_neighbors)
    X_res, y_res = resampler.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=test_size, random_state=seed, stratify=y_res
    )
    return SplitResult(X_train, X_test, y_train, y_test, pipeline="leaky", resampler=resampler_name)


def correct_pipeline_resampler(X, y, seed, resampler_name="smote", k_neighbors=5, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    resampler = _make_resampler(resampler_name, seed, k_neighbors)
    X_train_res, y_train_res = resampler.fit_resample(X_train, y_train)
    return SplitResult(X_train_res, X_test, y_train_res, y_test, pipeline="correct", resampler=resampler_name)
