"""
Runs the full experimental grid:

    dataset  x  classifier  x  {leaky, correct}  x  seed

For every combination: split/resample via the chosen pipeline, scale
features using a scaler fit on the TRAINING data only, train the
classifier, evaluate on the held-out test fold, and record every
metric needed for the leakage-inflation analysis.

Output: results_raw.csv, one row per run.

Usage:
    python run_experiment.py --n-seeds 30
    python run_experiment.py --n-seeds 5      # quick smoke test
"""

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from data_loaders import ALL_DATASETS
from pipelines import PIPELINES


def make_classifiers(seed):
    return {
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=seed),
        "xgboost": XGBClassifier(
            random_state=seed,
            eval_metric="logloss",
            n_estimators=300,
            verbosity=0,
        ),
    }


def evaluate(clf, X_train, X_test, y_train, y_test):
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)
    y_proba = clf.predict_proba(X_test_s)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_minority": f1_score(y_test, y_pred, pos_label=1, zero_division=0),
        "precision_minority": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
        "recall_minority": recall_score(y_test, y_pred, pos_label=1, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "auprc": average_precision_score(y_test, y_proba),
    }


def run(n_seeds: int, k_neighbors: int, test_size: float, out_path: str):
    rows = []
    t0 = time.time()

    for dataset_name, loader in ALL_DATASETS.items():
        X, y, meta = loader()

        for clf_name in make_classifiers(0).keys():
            for pipeline_name, pipeline_fn in PIPELINES.items():
                for seed in range(n_seeds):
                    split = pipeline_fn(X, y, seed=seed, k_neighbors=k_neighbors, test_size=test_size)
                    clf = make_classifiers(seed)[clf_name]
                    metrics = evaluate(clf, split.X_train, split.X_test, split.y_train, split.y_test)

                    rows.append({
                        "dataset": dataset_name,
                        "classifier": clf_name,
                        "pipeline": pipeline_name,
                        "seed": seed,
                        "k_neighbors": k_neighbors,
                        "n_samples": meta["n_samples"],
                        "n_features": meta["n_features"],
                        "imbalance_ratio": meta["imbalance_ratio"],
                        **metrics,
                    })

        print(f"[{time.time()-t0:6.1f}s] finished dataset: {dataset_name}")

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=30)
    parser.add_argument("--k-neighbors", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--out", type=str, default="results_raw.csv")
    args = parser.parse_args()

    run(args.n_seeds, args.k_neighbors, args.test_size, args.out)
