"""
Two figures for the Results section, built from results_raw.csv and
delta_table.csv.

fig1_distributions.png : per dataset, F1 (minority) distribution
                         across seeds, leaky vs correct, one subplot
                         per classifier.
fig2_inflation_bars.png: mean delta (leaky - correct) in F1, one bar
                         per dataset x classifier, sorted descending.

Usage:
    python make_plots.py --raw results_raw.csv --delta delta_table.csv
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_distributions(raw: pd.DataFrame, out_path: str, metric: str = "f1_minority"):
    datasets = sorted(raw.dataset.unique())
    classifiers = sorted(raw.classifier.unique())

    fig, axes = plt.subplots(len(datasets), len(classifiers), figsize=(4 * len(classifiers), 3.2 * len(datasets)), squeeze=False)

    for i, dataset in enumerate(datasets):
        for j, clf in enumerate(classifiers):
            ax = axes[i][j]
            sub = raw[(raw.dataset == dataset) & (raw.classifier == clf)]
            leaky_vals = sub[sub.pipeline == "leaky"][metric]
            correct_vals = sub[sub.pipeline == "correct"][metric]
            ax.boxplot([leaky_vals, correct_vals], tick_labels=["leaky", "correct"])
            ax.set_title(f"{dataset}\n{clf}", fontsize=9)
            if j == 0:
                ax.set_ylabel(metric)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_inflation_bars(delta_table: pd.DataFrame, out_path: str, metric: str = "f1_minority"):
    sub = delta_table[delta_table.metric == metric].copy()
    sub["label"] = sub["dataset"] + " / " + sub["classifier"]
    sub = sub.sort_values("mean_delta", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 0.45 * len(sub) + 1))
    colors = ["#c0392b" if v > 0 else "#2980b9" for v in sub["mean_delta"]]
    ax.barh(sub["label"], sub["mean_delta"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(f"mean delta ({metric}): leaky - correct")
    ax.set_title("Leakage-driven metric inflation by dataset x classifier")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=str, default="results_raw.csv")
    parser.add_argument("--delta", type=str, default="delta_table.csv")
    args = parser.parse_args()

    raw = pd.read_csv(args.raw)
    delta_table = pd.read_csv(args.delta)

    plot_distributions(raw, "fig1_distributions.png")
    plot_inflation_bars(delta_table, "fig2_inflation_bars.png")
