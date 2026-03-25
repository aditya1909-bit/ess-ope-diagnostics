# ess-ope-diagnostics

Goal: show that ESS is useful for low-bias importance-weighted estimators (`IS`, oracle `DR`) but is not a universal reliability diagnostic across biased estimators (`DM`, `FQE`). The benchmark now centers on ESS, bias/variance, and CI calibration in finite-horizon discrete MDPs.

## What This Repo Does
This project builds controlled `RandomMDP` and `chain_bandit` sweeps with exact dynamic-programming ground truth. The focused study compares four estimators:
- `IS-PDIS`
- oracle `DR`
- `DM` (`dm_tabular`)
- `FQE` (`fqe_linear`)

It tests three questions:
- whether ESS predicts total error for the unbiased side (`IS`, `DR`)
- whether that breaks for the biased side (`DM`, `FQE`)
- whether estimator-level confidence intervals are calibrated and cross-comparable

## Repo Flow (Recommended)
1. Run benchmark sweep + plot generation:
```bash
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python experiments/run_benchmark.py --config configs/sweeps/random_mdp_ci_ultra.yaml --interval-mode both --ci-level 0.95 --ci-bootstrap-samples 100 --estimators is_pdis,dr_oracle,dm_tabular,fqe_linear --num-workers -1 --mp-chunksize 1
```
2. Re-analyze any existing results:
```bash
PYTHONPATH=src .venv/bin/python experiments/analyze_results.py --results results/latest_random_mdp/sweep_results.parquet --output-dir results/latest_random_mdp/figures --interval-mode both --ci-level 0.95 --estimators is_pdis,dr_oracle,dm_tabular,fqe_linear
```

## Notebook Workflow
- `notebooks/02_interpret_results.ipynb`: load `results/latest_random_mdp` or `results/latest_chain_bandit`, regenerate figures, inspect report

## Key Output Files
After a run, see `results/<timestamp>_<name>/`:
- `sweep_results.parquet` / `sweep_results.csv`
- `figures/benchmark_fig1_ess_vs_error_by_estimator.*`
- `figures/benchmark_fig2_same_ess_different_error.*`
- `figures/benchmark_fig3_ess_changes_error_stability.*`
- `figures/benchmark_fig4_fan_estimate_vs_ess.*`
- `figures/benchmark_fig5_mean_variance_sensitivity.*` (for chain-bandit sweeps)
- `figures/benchmark_fig6_ess_bias_variance_mse.*`
- `figures/benchmark_fig7_matched_ess_risk_curves.*`
- `figures/benchmark_fig8_ci_coverage_width.*`
- `figures/benchmark_fig9_ci_method_comparison.*`
- `figures/benchmark_fig10_ess_story_dashboard.*`
- `figures/benchmark_fig11_ci_story_dashboard.*`
- `figures/benchmark_report.csv`
- `figures/estimator_summary.csv` (global estimator stats + bootstrap CI for ESS-error correlations)
- `figures/condition_summary.csv` (per `(alpha,beta,K)` error means + CI)
- `figures/bias_variance_summary.csv`
- `figures/ci_interval_summary.csv`
- `figures/ci_coverage_summary.csv`
- `figures/diagnostic_comparability.csv`
- `figures/chain_bandit_sensitivity_summary.csv` (how reward mean scale, reward gap, reward variance, and transition strength move error and ESS)
- `figures/paper_claims.csv` (claim-by-claim CI-aware verdicts)
- `figures/paper_claim_summary.csv` (supported / inconclusive / not_supported breakdown)

`results/latest_runs.json` is tracked in git and records the latest overall run plus the latest run for each benchmark family. The runtime symlinks `results/latest`, `results/latest_random_mdp`, and `results/latest_chain_bandit` are still created locally for convenience.

## Sweep Intensity Controls
Use these knobs in sweep YAML for stronger evidence:
- `env_repeats`: independent environment draws per `(seed, beta)`
- `policy_repeats`: independent policy-pair draws per environment
- `dataset_repeats`: independent offline datasets per condition
- `num_workers`: process-level parallelism for condition evaluation (macOS-safe `spawn` mode)
- `mp_chunksize`: number of condition tasks dispatched per worker chunk

Total rows:
`len(seeds) * len(betas) * len(alphas) * len(dataset_sizes) * env_repeats * policy_repeats * dataset_repeats`

Profiles:
- `configs/sweeps/random_mdp_ci_focused.yaml`: smaller ESS/CI sweep for fast iteration
- `configs/sweeps/random_mdp_ci_ultra.yaml`: primary heavy `RandomMDP` study
- `configs/sweeps/chain_bandit_ci_focused.yaml`: smaller confirmation study on `chain_bandit`

## Apple Silicon (Mac) Efficiency
- Parallelism is enabled via sweep config (`num_workers`, `mp_chunksize`) using multiprocessing `spawn`, which is the stable mode on macOS.
- Start with `num_workers: 8` on M-series laptops; increase gradually if memory allows.
- To avoid BLAS thread oversubscription with multiprocessing, run with:
```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python experiments/run_benchmark.py --config configs/sweeps/random_mdp_ci_ultra.yaml
```

Paper claims mode is enabled by default in `run_benchmark.py` and `analyze_results.py`.
Disable with `--no-paper-mode` if you only want figures + base summaries.

## Estimators
- `IS-PDIS`
- oracle `DR`
- `DM` (`dm_tabular`)
- `FQE` (`fqe_linear`)

## Environments
- `RandomMDP`: sparse transitions + misspecification knob (`beta`)
- `Gridworld`: interpretable baseline
- `ChainBanditEnv`: layered chain of bandit-like decisions; supports `reward_only` and `transitional` variants plus separate reward mean and reward variance sweeps

## Chain Bandit Benchmark
Use the chain benchmark when you want a simpler sequential testbed than a fully random MDP:
```bash
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python experiments/run_benchmark.py --config configs/sweeps/chain_bandit_ci_focused.yaml --interval-mode both --ci-level 0.95 --ci-bootstrap-samples 60 --estimators is_pdis,dr_oracle,dm_tabular,fqe_linear --num-workers -1 --mp-chunksize 1
```

Main chain-bandit sweep knobs:
- `transition_strengths`: how strongly actions affect the next state
- `reward_mean_scales`: overall signal level in reward means
- `reward_gaps`: separation between the preferred arm and the others
- `reward_stds`: reward noise level, independent of the reward means
- `chain_variants`: `reward_only` keeps transitions action-independent; `transitional` makes the problem genuinely sequential

## Reproducibility
Each sweep run stores:
- config YAML
- git commit hash (or `unknown` when unavailable)
- seeds and per-condition `env_seed` / `policy_seed` / `dataset_seed`
- raw table + generated figures

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest -q
```

## Citation / License
