# ess-ope-diagnostics

This repo now contains only the v2 OPE uncertainty-diagnostics framework.

## What It Does
The v2 pipeline studies whether ESS and confidence-interval diagnostics track true uncertainty in offline policy evaluation across:
- contextual bandits
- short-horizon tabular MDPs
- rare-event / sparse-reward MDPs

Main estimators:
- `is`
- `wis`
- `pdis`
- `dr`
- `wdr`
- `fqe_linear`

Optional reference estimators can be enabled in config:
- `dr_oracle`
- `fqe_tabular`

Core outputs:
- raw replicate-level results
- Table A estimator summaries
- Table B diagnostic-quality summaries
- calibration summaries
- failure-prediction summaries
- cross-estimator ranking tables
- paper figures under `v2_fig*.png`

## Main Workflow
Run one small phase:
```bash
PYTHONPATH=src .venv/bin/python experiments/run_v2_phase.py --config configs/v2/tiny_bandit_phase.yaml
```

Run the full suite:
```bash
PYTHONPATH=src .venv/bin/python experiments/run_v2_suite.py --config configs/v2/paper_suite.yaml
```

In the main phase configs, `num_workers: -1` means "use all detected CPU cores".

Re-analyze an existing replicate table:
```bash
PYTHONPATH=src .venv/bin/python experiments/analyze_v2_results.py \
  --results results/latest/replicate_results.csv \
  --output-dir results/latest/paper_artifacts
```

## Configs
Primary configs:
- `configs/v2/bandit_phase.yaml`
- `configs/v2/tabular_phase.yaml`
- `configs/v2/rare_event_phase.yaml`
- `configs/v2/paper_suite.yaml`

Lighter and faster alternatives:
- `configs/v2/paper_suite_practical.yaml`
- `configs/v2/paper_suite_fast.yaml`

Smoke-test configs:
- `configs/v2/tiny_bandit_phase.yaml`
- `configs/v2/tiny_tabular_phase.yaml`
- `configs/v2/tiny_rare_event_phase.yaml`
- `configs/v2/paper_suite_tiny.yaml`

## Outputs
Each run writes a timestamped directory under `results/` with:
- `replicate_results.csv` / `replicate_results.parquet`
- `artifacts/` or `paper_artifacts/`

The artifact directory contains:
- `table_a_estimator_summary.csv`
- `table_b_diagnostic_quality.csv`
- `calibration_summary.csv`
- `failure_prediction_summary.csv`
- `cross_estimator_ranking.csv`
- `condition_variance_summary.csv`
- `diagnostic_correlation_summary.csv`
- `v2_fig*.png`

`results/latest` is refreshed to the newest run.

Recommended run variants:
- `paper_suite.yaml`: heavy full study
- `paper_suite_practical.yaml`: lighter paper-like run
- `paper_suite_fast.yaml`: materially faster iteration run

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest -q
```
