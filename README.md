# SMOTE Leakage Quantification — Experiment Code

Implements the two pipelines described in the paper plan and runs the
full dataset x classifier x pipeline x seed grid.

## Files

- `data_loaders.py` — loads the three clinical datasets (Pima Diabetes,
  Cleveland Heart Disease, Wisconsin Breast Cancer). Datasets are
  cached under `data/`. Add a fourth dataset by writing a loader with
  the same `(X, y, meta)` signature and registering it in
  `ALL_DATASETS`.
- `pipelines.py` — `leaky_pipeline` (SMOTE before split) and
  `correct_pipeline` (SMOTE after split, training fold only). This is
  the core experimental manipulation; everything else holds constant
  around it.
- `run_experiment.py` — runs the full grid, writes `results_raw.csv`
  (one row per dataset x classifier x pipeline x seed run).
- `analysis.py` — computes per-combination mean delta (leaky −
  correct), paired Wilcoxon signed-rank tests, Cliff's delta effect
  sizes, writes `delta_table.csv`, and runs the inflation-predictor
  OLS regression (delta ~ imbalance ratio + log(n) + n_features).
- `make_plots.py` — generates `fig1_distributions.png` (leaked vs
  correct metric spread per dataset/classifier) and
  `fig2_inflation_bars.png` (ranked mean inflation).

## Run order

```bash
pip install -r requirements.txt

# full run — 30 seeds per combination, ~10-15 min on CPU
python run_experiment.py --n-seeds 30

# or a fast check first
python run_experiment.py --n-seeds 5 --out results_smoketest.csv

python analysis.py --in results_raw.csv
python make_plots.py --raw results_raw.csv --delta delta_table.csv
```

## k-neighbors ablation

To check whether inflation magnitude depends on SMOTE's `k`, rerun
with different `--k-neighbors` values and compare `delta_table.csv`
across runs:

```bash
for k in 3 5 7 10; do
  python run_experiment.py --n-seeds 15 --k-neighbors $k --out results_k${k}.csv
done
```

Concatenate the resulting `results_k*.csv` files (they include a
`k_neighbors` column) before feeding them through `analysis.py` if you
want the ablation folded into one delta table.

## Known limitation in the current form

The inflation-predictor regression in `analysis.py` is descriptive
only with three datasets — the design matrix is close to saturated
(one row per dataset x classifier, only three distinct dataset
profiles), which `statsmodels` will flag as rank-deficient. Add more
datasets to `data_loaders.py` before treating the regression
coefficients as more than a plausibility check. This is exactly the
kind of limitation to state explicitly in the manuscript rather than
paper over.

## Verified

All scripts have been run end-to-end on a 3-seed smoke test; the
expected direction of the effect (leaky pipeline reports higher F1 /
ROC-AUC than correct pipeline) holds across all nine dataset x
classifier combinations already at that reduced scale.

## Citation

Code and cached data: https://github.com/mikehyderabad200-sys/smote-leakage-quantification
