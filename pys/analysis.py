"""
Turns results_raw.csv into the tables/statistics that go directly
into the Results section.

1. Per dataset x classifier x metric: mean(leaky) - mean(correct) = delta,
   plus a paired Wilcoxon signed-rank test across seeds and Cliff's
   delta effect size.
2. A single long-format table of all deltas (delta_table.csv).
3. An OLS regression of mean delta_f1 on dataset-level properties
   (imbalance ratio, log(n_samples), n_features) across the
   dataset x classifier combinations, to test whether inflation
   magnitude is predictable from dataset shape.

Usage:
    python analysis.py --in results_raw.csv
    python analysis.py --in results_k3.csv --out delta_table_k3.csv
"""

import argparse

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

METRICS = ["accuracy", "f1_minority", "precision_minority", "recall_minority", "roc_auc", "auprc"]


def cliffs_delta(x, y):
    """Effect size for paired non-normal data. Range [-1, 1]."""
    x, y = np.asarray(x), np.asarray(y)
    n = len(x)
    diffs = x[:, None] - y[None, :]
    return (np.sum(diffs > 0) - np.sum(diffs < 0)) / (n * n)


def build_delta_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["dataset", "classifier", "n_samples", "n_features", "imbalance_ratio"]
    # SMOTE's k is an experimental condition, not a nuisance column: keep it in
    # the grouping so a concatenated multi-k input yields one row per k rather
    # than silently averaging the ablation levels together.
    if "k_neighbors" in df.columns:
        group_cols = ["k_neighbors"] + group_cols

    for keys, g in df.groupby(group_cols):
        key = dict(zip(group_cols, keys))
        dataset = key["dataset"]
        classifier = key["classifier"]
        n_samples = key["n_samples"]
        n_features = key["n_features"]
        imbalance_ratio = key["imbalance_ratio"]
        leaky = g[g.pipeline == "leaky"].sort_values("seed")
        correct = g[g.pipeline == "correct"].sort_values("seed")

        # align on shared seeds so the paired test is valid
        shared_seeds = sorted(set(leaky.seed) & set(correct.seed))
        leaky = leaky[leaky.seed.isin(shared_seeds)]
        correct = correct[correct.seed.isin(shared_seeds)]

        for metric in METRICS:
            lv = leaky[metric].to_numpy()
            cv = correct[metric].to_numpy()
            delta = lv - cv

            try:
                stat, p = wilcoxon(lv, cv)
            except ValueError:
                # all differences zero, or n too small
                stat, p = np.nan, np.nan

            rows.append({
                "dataset": dataset,
                "classifier": classifier,
                "k_neighbors": key.get("k_neighbors", np.nan),
                "metric": metric,
                "n_samples": n_samples,
                "n_features": n_features,
                "imbalance_ratio": imbalance_ratio,
                "mean_leaky": lv.mean(),
                "mean_correct": cv.mean(),
                "mean_delta": delta.mean(),
                "std_delta": delta.std(ddof=1) if len(delta) > 1 else np.nan,
                "wilcoxon_stat": stat,
                "wilcoxon_p": p,
                "cliffs_delta": cliffs_delta(lv, cv),
                "n_seeds": len(shared_seeds),
            })

    return pd.DataFrame(rows)


def add_fdr_correction(delta_table: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """
    Benjamini-Hochberg FDR correction across every test in the table
    (all dataset x classifier x metric combinations at once — 54
    tests in the 3-dataset x 3-classifier x 6-metric grid). This is
    the multiple-comparisons correction the raw wilcoxon_p column
    does NOT have; report p_fdr / significant_fdr in the manuscript,
    not wilcoxon_p on its own.

    BH is used rather than Bonferroni because the six metrics within
    one dataset x classifier run are correlated (not independent
    tests), which makes Bonferroni needlessly conservative here.
    """
    delta_table = delta_table.copy()
    valid = delta_table["wilcoxon_p"].notna()

    p_fdr = np.full(len(delta_table), np.nan)
    significant = np.zeros(len(delta_table), dtype=bool)

    if valid.sum() > 0:
        _, p_corrected, _, _ = multipletests(
            delta_table.loc[valid, "wilcoxon_p"].to_numpy(), alpha=alpha, method="fdr_bh"
        )
        p_fdr[valid.to_numpy()] = p_corrected
        significant[valid.to_numpy()] = p_corrected < alpha

    delta_table["p_fdr_bh"] = p_fdr
    delta_table["significant_fdr_bh"] = significant
    return delta_table


def run_inflation_regression(delta_table: pd.DataFrame, metric: str = "f1_minority"):
    """
    Regresses mean_delta (for one metric) on dataset-level properties
    across the dataset x classifier combinations. With only a
    handful of datasets this is under-powered for real inference —
    treat coefficients as descriptive, and say so explicitly in the
    Limitations section. Add more datasets to give this real power.
    """
    sub = delta_table[delta_table.metric == metric].copy()
    sub["log_n_samples"] = np.log(sub["n_samples"])

    X = sub[["imbalance_ratio", "log_n_samples", "n_features"]]
    X = sm.add_constant(X)
    y = sub["mean_delta"]

    model = sm.OLS(y, X).fit()
    return model


def main(in_path: str, out_path: str = "delta_table.csv"):
    df = pd.read_csv(in_path)

    delta_table = build_delta_table(df)
    delta_table = add_fdr_correction(delta_table)
    delta_table.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(delta_table)} rows)\n")

    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 20)

    n_total = len(delta_table)
    n_sig_raw = (delta_table["wilcoxon_p"] < 0.05).sum()
    n_sig_fdr = delta_table["significant_fdr_bh"].sum()
    print(f"Significant at raw p<0.05: {n_sig_raw}/{n_total}")
    print(f"Significant after BH-FDR correction (alpha=0.05): {n_sig_fdr}/{n_total}\n")

    print("=== Significant tests (BH-FDR corrected) by dataset ===")
    print(
        delta_table.groupby("dataset")["significant_fdr_bh"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "n_significant_fdr", "count": "n_tests"})
        .to_string()
    )

    print("\n=== Mean inflation (leaky - correct) by dataset x classifier, F1 (minority) ===")
    print(
        delta_table[delta_table.metric == "f1_minority"]
        .sort_values("mean_delta", ascending=False)
        [["dataset", "classifier", "mean_delta", "wilcoxon_p", "p_fdr_bh", "significant_fdr_bh", "cliffs_delta", "n_seeds"]]
        .to_string(index=False)
    )

    print("\n=== Inflation-predictor regression (F1 delta ~ imbalance ratio + log(n) + n_features) ===")
    model = run_inflation_regression(delta_table, metric="f1_minority")
    print(model.summary())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", type=str, default="results_raw.csv")
    parser.add_argument("--out", dest="out_path", type=str, default="delta_table.csv")
    args = parser.parse_args()
    main(args.in_path, args.out_path)