# ess-ope-diagnostics

Paper-specific simulation repo for the revised ESS story:

- within IS-like estimators, weight-based `WESS` tracks degeneracy but not full uncertainty
- even there, changing reward variance can move estimator variance, absolute error, and CI width while `WESS` stays nearly fixed
- across estimator families, shared `WESS` is not a portable comparison language
- interval width and empirical coverage are definable across families, so they are the more defensible common reporting layer

## Simulation Structure
The paper-facing study is organized around three domains and five paper experiments:

1. `experiment_1`: Domain A bandit reward-variance intervention
2. `experiment_2`: Domain A within-family `WESS` vs absolute-error relationship
3. `experiment_3`: Domain B short-horizon tabular MDP cross-family comparison
4. `experiment_4`: Domain C long-horizon tabular MDP mismatch stress test
5. `experiment_5`: FQE bootstrap calibration case study

The domains are:

- Domain A: contextual bandit with `S=10`, `A=5`
- Domain B: short-horizon tabular MDP with `S=30`, `A=3`, `H=5`
- Domain C: long-horizon tabular MDP with `S=40`, `A=3`, `H=20`

## Estimators
The main estimator set is:

- `is`
- `snis`
- `pdis`
- `dm`
- `dr`
- `mrdr`
- `fqe`

`WDR` is intentionally not in the default paper suite.

## Global Protocol
The repo now follows one paper-wide reporting protocol:

- exact ground truth for every simulation condition
- shared dataset seeds across estimators within each condition
- `90%` confidence intervals in the main study
- sample bootstrap for bandits
- episode bootstrap for MDPs
- repeated-dataset summaries based on:
  - mean `WESS`
  - mean OPE absolute error
  - empirical estimator variance
  - mean CI width
  - empirical CI coverage
  - Spearman(`WESS`, absolute error)
  - Spearman(CI width, absolute error)

## Default Experiment Grids
The paper configs use explicit numeric controls rather than labels like `low` or `high`.

- Domain A reward-variance scales: `0.1, 0.25, 0.5, 1.0, 2.0, 4.0`
- Domain B/C mismatch values: `alpha in {0.0, 0.2, 0.4, 0.6, 0.8}`
- Main sample sizes: `100, 300, 1000`
- Repetitions:
  - `paper_full`: `500`
  - `paper_practical`: `200`
  - `paper_tiny`: smoke-test scale

## Figure Lineup
The artifact generator is aligned with the revised paper:

- `figure_1.png`: bandit reward-variance sanity check
- `figure_2.png`: within-family `WESS` vs absolute error scatter
- `figure_3.png`: grouped Spearman bars for `WESS` and CI width
- `figure_4.png`: long-horizon mismatch stress test
- `figure_5.png`: empirical `90%` coverage vs sample size
- `figure_6.png`: mean `90%` CI width vs sample size
- `figure_7.png`: FQE bootstrap calibration
- `appendix_figure_a1.png`: cross-family `WESS` vs absolute error scatter

## Main Commands
Use the repo venv and set both `PYTHONPATH` and `MPLCONFIGDIR`:

```bash
source .venv/bin/activate
export PYTHONPATH=src
export MPLCONFIGDIR=/tmp/matplotlib
```

Run one experiment:

```bash
python experiments/run_experiment.py --config configs/study/experiment_3.yaml
```

Run the practical study:

```bash
python experiments/run_study.py --config configs/study/paper_practical.yaml
```

Run the full study:

```bash
python experiments/run_study.py --config configs/study/paper_full.yaml
```

Re-analyze saved replicate results:

```bash
python experiments/analyze_results.py \
  --results results/latest/artifacts/replicate_results.csv
```

## Outputs
Each run writes a timestamped directory under `results/` containing:

- `artifacts/replicate_results.csv`
- `artifacts/point_estimates.csv`
- `artifacts/condition_summary.csv`
- `artifacts/table_1_main_summary.csv`
- `artifacts/table_2_diagnostic_usefulness.csv`
- `artifacts/figure_1.png` through `artifacts/figure_7.png`
- `artifacts/appendix_figure_a1.png`
- `config.yaml`
- `metadata.json`

`results/latest` is refreshed to the newest run.

## Narrative Order
The simulation section is meant to read in this order:

1. Bandit reward-variance intervention: `WESS` misses reward-driven uncertainty.
2. Within-family scatter: `WESS` still carries partial information about degeneracy.
3. Short-horizon cross-family comparison: shared `WESS` does not port across estimator families.
4. Long-horizon stress test: IS-style methods degrade sharply as mismatch grows.
5. Interval-based comparison: width and coverage are comparable outputs across families.
6. FQE case study: bootstrap intervals can become meaningful in a family-specific setting.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=src .venv/bin/pytest -q
```
