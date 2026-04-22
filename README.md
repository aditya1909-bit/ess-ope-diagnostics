# ess-ope-diagnostics

Paper-specific simulation repo for testing when weight-based ESS tracks uncertainty in offline policy evaluation, when it fails, and how interval width and empirical coverage compare across estimator families.

## Study Structure
The primary repo surface is a fixed 5-experiment study:

1. `experiment_1`: bandit same-weights / different-reward-variance sanity check
2. `experiment_2`: bandit same-mismatch / different-reward-mean structure
3. `experiment_3`: short-horizon tabular MDP cross-family comparison
4. `experiment_4`: long-horizon mismatch stress test
5. `experiment_5`: FQE bootstrap calibration on short and long MDP settings

Environments:
- contextual bandit: `S=20`, `A=4`
- short tabular MDP: `S=30`, `A=4`, `H=5`
- long tabular MDP: `S=50`, `A=4`, `H=20`

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

## Main Commands
Run one experiment:
```bash
PYTHONPATH=src .venv/bin/python experiments/run_experiment.py --config configs/study/experiment_3.yaml
```

Run the full paper study:
```bash
PYTHONPATH=src .venv/bin/python experiments/run_study.py --config configs/study/paper_practical.yaml
```

Re-analyze saved replicate results:
```bash
PYTHONPATH=src .venv/bin/python experiments/analyze_results.py \
  --results results/latest/artifacts/replicate_results.csv
```

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

## Narrative Order
The analysis and figures follow the paper narrative directly:
- bandit sanity checks
- short-horizon cross-family comparison
- long-horizon stress test
- FQE interval calibration

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
.venv/bin/pytest -q
```
