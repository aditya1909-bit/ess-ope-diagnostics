# ess-ope-diagnostics

Paper-specific simulation repo for testing when weight-based ESS tracks uncertainty in offline policy evaluation, when it fails, and how interval width and empirical coverage compare across estimator families.

The current study results support the main thesis:
- within IS-like estimators, `WESS` tracks weight degeneracy but misses other uncertainty sources
- across estimator families, shared `WESS` is not a meaningful universal uncertainty diagnostic
- interval width is a comparable output across methods, but empirical coverage remains estimator-specific and often poor

The full study also now supports the revised secondary claims:
- `experiment_2` cleanly shows nearly unchanged `WESS` with materially different RMSE and CI width
- FQE bootstrap calibration works in the easier short-horizon positive case and in the easier long-horizon stress case only once sample size is large enough

## Study Structure
The repo surface is a fixed 5-experiment study:

1. `experiment_1`: bandit same-weights / different-reward-variance sanity check
2. `experiment_2`: bandit same-mismatch / different-reward-mean structure
3. `experiment_3`: short-horizon tabular MDP cross-family comparison
4. `experiment_4`: long-horizon mismatch stress test
5. `experiment_5`: FQE bootstrap calibration with a short positive case and a long stress case

Environments:
- contextual bandit: `S=20`, `A=4`
- short tabular MDP: `S=30`, `A=4`, `H=5`
- long tabular MDP stress test: `S=50`, `A=4`, `H=20`
- long FQE stress-calibration case: easier long-horizon variant with `H=12`, denser rewards, and lower mismatch

Estimator set:
- `is`
- `snis`
- `pdis`
- `dm`
- `dr`
- `mrdr`
- `fqe`

Default study conventions:
- exact dynamic-programming truth for all environments
- shared dataset seeds across estimators within each condition
- episode-level bootstrap only
- 90% as the primary CI level, 95% as secondary
- Spearman correlations for diagnostic-strength summaries
- parallel execution on all detected logical CPUs by default

## What The Full Study Shows
These are the main takeaways from the current `paper_full` run:

- `experiment_1`: mean `WESS` stays nearly constant across reward-variance regimes while RMSE and interval width increase, which is the cleanest direct evidence that equal `WESS` does not imply equal uncertainty.
- `experiment_2`: after the redesign, `smooth` and `heterogeneous` reward structures have nearly identical `WESS`, but `heterogeneous` produces clearly larger RMSE and wider intervals.
- `experiment_3`: `DM` is best on the main benchmark, `DR` is next, `PDIS` is the strongest IS-family baseline, and shared `WESS` does not recover that cross-family ranking.
- `experiment_4`: as mismatch increases, `WESS` collapses and the IS family degrades sharply, while `DM` remains comparatively stable.
- `experiment_5`: FQE bootstrap intervals calibrate well in the short positive case by moderate-to-large sample sizes and in the easier long stress case only at large sample size. The repo does not support a claim that FQE is uniformly well-calibrated in hard long-horizon settings.

The README therefore intentionally does **not** claim that interval width is always a strong per-replicate error predictor or that all bootstrap intervals are well-calibrated. The evidence is stronger and narrower than that.

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

The study runner now shows:
- an `Overall Study` progress bar with ETA
- a per-experiment progress bar with ETA

## Expected Runtime
Approximate wall-clock times on the current machine after parallelization:
- `paper_tiny`: about 20-30 seconds
- `paper_practical`: several hours
- `paper_full`: a few hours to low double-digit hours, depending on system load

The dominant cost is bootstrap refitting of the model-based estimators in `experiment_3` to `experiment_5`.

## Config Tiers
- `configs/study/paper_full.yaml`: 500 replicates, 500 bootstraps
- `configs/study/paper_practical.yaml`: 200 replicates, 200 bootstraps
- `configs/study/paper_tiny.yaml`: smoke-test scale

Individual experiments live under `configs/study/experiment_*.yaml`.

## Outputs
Each run writes a timestamped directory under `results/` containing:
- `artifacts/replicate_results.csv`
- `artifacts/condition_summary.csv`
- `artifacts/table_1_main_summary.csv`
- `artifacts/table_2_diagnostic_usefulness.csv`
- `artifacts/figure_1.png` through `artifacts/figure_8.png`
- `config.yaml`
- `metadata.json`

`results/latest` is refreshed to the newest run.

The most important outputs are:
- `table_1_main_summary.csv`: main benchmark bias, variance, RMSE, mean `WESS`, mean CI width, and coverage
- `table_2_diagnostic_usefulness.csv`: Spearman diagnostic summaries
- `replicate_results.csv`: the raw table used for all figures and summaries

## Narrative Order
The analysis and figures follow the paper narrative directly:
- bandit sanity checks
- short-horizon cross-family comparison
- long-horizon stress test
- FQE interval calibration

For the current results, the constructive interpretation should be:
- `WESS` is not a universal cross-family uncertainty language
- interval outputs should be judged by empirical coverage and width, estimator by estimator
- FQE intervals can be meaningful, but only in regimes where the estimator itself is not badly biased

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=src .venv/bin/pytest -q
```
