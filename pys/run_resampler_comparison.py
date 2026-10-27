"""
Extends the core leaky-vs-correct comparison to ADASYN and Borderline-SMOTE,
alongside the original SMOTE, to test whether leakage-driven inflation is
specific to SMOTE or general to interpolation-based over-samplers.

Usage:
    python run_resampler_comparison.py --n-seeds 15
"""

import argparse
import time

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from data_loaders import ALL_DATASETS
from pipelines_multi_resampler import leaky_pipeline_resampler, correct_pipeline_resampler, RESAMPLERS


def make_classifiers(seed):
    return {
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=seed),
        "xgboost": XGBClassifier(random_state=seed, eval_metric="logloss", n_estimators=300, verbosity=0),
    }


def evaluate(clf, X_train, X_test, y_train, y_test):
    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)
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


def run(n_seeds, k_neighbors, test_size, out_path):
    rows = []
    t0 = time.time()
    pipelines = {"leaky": leaky_pipeline_resampler, "correct": correct_pipeline_resampler}

    for resampler_name in RESAMPLERS.keys():
        for dataset_name, loader in ALL_DATASETS.items():
            X, y, meta = loader()
            for clf_name in make_classifiers(0).keys():
                for pipeline_name, pipeline_fn in pipelines.items():
                    for seed in range(n_seeds):
                        try:
                            split = pipeline_fn(X, y, seed=seed, resampler_name=resampler_name,
                                                 k_neighbors=k_neighbors, test_size=test_size)
                        except ValueError as e:
                            # ADASYN/BorderlineSMOTE can fail on tiny minority folds; skip and log
                            print(f"SKIP {resampler_name}/{dataset_name}/{clf_name}/{pipeline_name}/seed{seed}: {e}")
                            continue
                        clf = make_classifiers(seed)[clf_name]
                        metrics = evaluate(clf, split.X_train, split.X_test, split.y_train, split.y_test)
                        rows.append({
                            "resampler": resampler_name, "dataset": dataset_name, "classifier": clf_name,
                            "pipeline": pipeline_name, "seed": seed, "k_neighbors": k_neighbors,
                            "n_samples": meta["n_samples"], "n_features": meta["n_features"],
                            "imbalance_ratio": meta["imbalance_ratio"], **metrics,
                        })
            print(f"[{time.time()-t0:6.1f}s] finished {resampler_name} / {dataset_name}")

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=15)
    parser.add_argument("--k-neighbors", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--out", type=str, default="results_resampler_comparison.csv")
    args = parser.parse_args()
    run(args.n_seeds, args.k_neighbors, args.test_size, args.out)
