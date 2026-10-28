# Resampling Before the Split: Quantifying Leakage-Driven Inflation in SMOTE-Based Clinical Classifiers

## Experiment Code

This repository implements the experimental pipelines used to investigate how applying SMOTE before train-test splitting can inflate reported clinical classification performance.

## Files

- `data_loaders.py`: Loads clinical datasets, including Pima Diabetes, Cleveland Heart Disease, and Wisconsin Breast Cancer. Cached datasets are stored under `data/`.
- `pipelines.py`: Implements the leaky pipeline (SMOTE before splitting) and the correct pipeline (SMOTE applied only to the training fold).
- `run_experiment.py`: Runs the experiment grid and writes raw results to CSV.
- `analysis.py`: Computes mean performance differences between the leaky and correct pipelines, paired Wilcoxon signed-rank tests, Cliff's delta effect sizes, and an exploratory OLS regression.
- `make_plots.py`: Generates performance distribution and inflation comparison figures.

## Requirements

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Running the Experiments

Run a quick test:

```bash
python run_experiment.py --n-seeds 5 --out results_smoketest.csv
```

Run the full experiment with 30 seeds:

```bash
python run_experiment.py --n-seeds 30
```

Analyze the results and generate plots:

```bash
python analysis.py --in results_raw.csv
python make_plots.py --raw results_raw.csv --delta delta_table.csv
```

Run these commands from the directory containing the relevant Python scripts, or adjust the paths if your scripts are stored in the `pys/` directory.

## SMOTE k-Neighbors Ablation

To investigate how SMOTE's `k_neighbors` parameter affects performance inflation, run experiments with different values of `k`.

Example for Windows Command Prompt:

```cmd
python run_experiment.py --n-seeds 15 --k-neighbors 3 --out results_k3.csv
python run_experiment.py --n-seeds 15 --k-neighbors 5 --out results_k5.csv
python run_experiment.py --n-seeds 15 --k-neighbors 7 --out results_k7.csv
python run_experiment.py --n-seeds 15 --k-neighbors 10 --out results_k10.csv
```

Compare the resulting CSV files to assess how the size of the performance difference varies with `k_neighbors`.

## Statistical Limitations

The inflation-predictor regression is exploratory when based on only three distinct dataset profiles. Its coefficients may be unstable or poorly identified. Additional datasets are needed before drawing strong conclusions from this regression.

## Reproducibility

Record the dataset versions, random seeds, classifier settings, and SMOTE parameters used for each experiment. Report the results of the full experiment rather than relying exclusively on smoke-test results.

