# ess-ope-diagnostics

Goal: Show that IS-ESS (weight degeneracy metric) is not a universal reliability metric for OPE estimators by demonstrating weak/unstable ESS-error relationships and controlled counterexamples for DM/FQE/MRDR in finite-horizon discrete MDPs.

## What This Repo Does
This project builds controlled RandomMDP sweeps with exact dynamic-programming ground truth. It compares IS-family estimators against DM/FQE/MRDR and tests two core counterexample patterns:
- same/similar ESS but very different DM/FQE/MRDR error
- ESS changes a lot while DM/FQE error changes little in learnable regimes

## Repo Flow (Recommended)
1. Run benchmark sweep + plot generation:
```bash
python experiments/run_benchmark.py --config configs/sweeps/random_mdp_ultra.yaml
```
2. Re-analyze any existing results:
```bash
python experiments/analyze_results.py --results results/latest/sweep_results.parquet --output-dir results/latest/figures
```
3. Single-condition debugging run:
```bash
python experiments/run_single.py --alpha 0.4 --beta 0.5 --episodes 200 --seed 0
```

## Notebook Workflow
- `notebooks/01_run_benchmark.ipynb`: run a sweep and produce benchmark figures/report
- `notebooks/02_interpret_results.ipynb`: load `results/latest`, regenerate figures, inspect report

## Key Output Files
After a run, see `results/<timestamp>_<name>/`:
- `sweep_results.parquet` / `sweep_results.csv`
- `figures/benchmark_fig1_ess_vs_error_by_estimator.*`
- `figures/benchmark_fig2_same_ess_different_error.*`
- `figures/benchmark_fig3_ess_changes_error_stability.*`
- `figures/benchmark_fig4_fan_estimate_vs_ess.*`
- `figures/benchmark_fig5_mean_variance_sensitivity.*` (for chain-bandit sweeps)
- `figures/benchmark_report.csv`
- `figures/estimator_summary.csv` (global estimator stats + bootstrap CI for ESS-error correlations)
- `figures/condition_summary.csv` (per `(alpha,beta,K)` error means + CI)
- `figures/chain_bandit_sensitivity_summary.csv` (how reward mean scale, reward gap, reward variance, and transition strength move error and ESS)
- `figures/paper_claims.csv` (claim-by-claim CI-aware verdicts)
- `figures/paper_claim_summary.csv` (supported / inconclusive / not_supported breakdown)

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
- `configs/sweeps/random_mdp_baseline.yaml`: quick checks
- `configs/sweeps/random_mdp_robust.yaml`: strong default benchmark
- `configs/sweeps/random_mdp_intensive.yaml`: 20x `robust` row count for tighter estimates
- `configs/sweeps/random_mdp_ultra.yaml`: higher-than-intensive run for maximum stability
- `configs/sweeps/chain_bandit_baseline.yaml`: layered chain-of-bandits benchmark with explicit reward mean/variance sweeps
- `configs/sweeps/chain_bandit_focused.yaml`: notebook-friendly chain-bandit run focused on mean/variance effects with full-machine multiprocessing defaults for this M3 Pro setup

## Apple Silicon (Mac) Efficiency
- Parallelism is enabled via sweep config (`num_workers`, `mp_chunksize`) using multiprocessing `spawn`, which is the stable mode on macOS.
- Start with `num_workers: 8` on M-series laptops; increase gradually if memory allows.
- To avoid BLAS thread oversubscription with multiprocessing, run with:
```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python experiments/run_benchmark.py --config configs/sweeps/random_mdp_ultra.yaml
```

Paper claims mode is enabled by default in `run_benchmark.py` and `analyze_results.py`.
Disable with `--no-paper-mode` if you only want figures + base summaries.

## Estimators
- IS / WIS / PDIS
- DM (tabular model-based)
- FQE (tabular + linear)
- DR
- MRDR-inspired weighted linear DR pipeline

## Environments
- `RandomMDP`: sparse transitions + misspecification knob (`beta`)
- `Gridworld`: interpretable baseline
- `ChainBanditEnv`: layered chain of bandit-like decisions; supports `reward_only` and `transitional` variants plus separate reward mean and reward variance sweeps

## Chain Bandit Benchmark
Use the chain benchmark when you want a simpler sequential testbed than a fully random MDP:
```bash
python experiments/run_benchmark.py --config configs/sweeps/chain_bandit_baseline.yaml
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
