"""
Dataset loaders for the SMOTE-leakage quantification study.

Three clinical binary-classification datasets, chosen to vary in
imbalance ratio, sample size, and feature count:

    - Pima Indians Diabetes   (n=768,  8 features,  ~35% positive)
    - Cleveland Heart Disease (n=303, 13 features,  ~46% positive)
    - Wisconsin Breast Cancer (n=569, 30 features,  ~37% positive)

Each loader returns (X, y) as numpy arrays, plus a small metadata
dict (n_samples, n_features, imbalance_ratio) used later to model
inflation as a function of dataset properties.

Add a fourth dataset by writing a loader with the same signature and
registering it in ALL_DATASETS at the bottom of this file.
"""

import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _metadata(y: np.ndarray, n_features: int) -> dict:
    n = len(y)
    n_pos = int(y.sum())
    n_neg = n - n_pos
    minority = min(n_pos, n_neg)
    majority = max(n_pos, n_neg)
    return {
        "n_samples": n,
        "n_features": n_features,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "imbalance_ratio": majority / minority,  # >=1, higher = more imbalanced
    }


def load_pima():
    path = os.path.join(DATA_DIR, "pima_diabetes.csv")
    df = pd.read_csv(path)
    X = df.drop(columns=["Outcome"]).to_numpy(dtype=float)
    y = df["Outcome"].to_numpy(dtype=int)
    return X, y, _metadata(y, X.shape[1])


def load_cleveland():
    path = os.path.join(DATA_DIR, "cleveland_heart.csv")
    df = pd.read_csv(path, encoding="utf-8-sig")
    # 'target' is already binary (0 = no disease, 1 = disease) in this mirror
    X = df.drop(columns=["target"]).to_numpy(dtype=float)
    y = df["target"].to_numpy(dtype=int)
    return X, y, _metadata(y, X.shape[1])


def load_wisconsin_breast_cancer():
    data = load_breast_cancer()
    X = data.data.astype(float)
    # sklearn encodes 0=malignant, 1=benign; flip so 1 = disease-positive
    # (malignant), matching the "positive = condition present" convention
    # used for the other two datasets
    y = (data.target == 0).astype(int)
    return X, y, _metadata(y, X.shape[1])


ALL_DATASETS = {
    "pima_diabetes": load_pima,
    "cleveland_heart": load_cleveland,
    "wisconsin_breast_cancer": load_wisconsin_breast_cancer,
}


if __name__ == "__main__":
    for name, loader in ALL_DATASETS.items():
        X, y, meta = loader()
        print(name, meta)
