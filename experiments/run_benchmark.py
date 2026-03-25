#!/usr/bin/env python
from __future__ import annotations

import argparse

from ess_ope.evaluation.sweep import SweepConfig, run_sweep
from ess_ope.evaluation.reporting import generate_run_artifacts
from ess_ope.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ESS benchmark sweep and generate comparison figures for IS/DR/DM/FQE"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/sweeps/random_mdp_ci_ultra.yaml",
        help="Sweep YAML config",
    )
    parser.add_argument(
        "--fixed-alpha",
        type=float,
        default=None,
        help="Optional alpha slice for same-ESS counterexample panels",
    )
    parser.add_argument(
        "--fixed-beta",
        type=float,
        default=0.0,
        help="Beta slice for ESS-changes-vs-error-stability panel",
    )
    parser.add_argument(
        "--fig-subdir",
        type=str,
        default="figures",
        help="Figure output subdirectory inside run dir",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=16000,
        help="Bootstrap samples for correlation confidence intervals in summary tables",
    )
    parser.add_argument(
        "--bootstrap-max-points",
        type=int,
        default=200000,
        help="Max points used for bootstrap correlation CI computation",
    )
    parser.add_argument(
        "--paper-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write CI-aware paper claims tables (paper_claims.csv, paper_claim_summary.csv).",
    )
    parser.add_argument(
        "--ci-level",
        type=float,
        default=0.95,
        help="Confidence level used for estimator interval summaries.",
    )
    parser.add_argument(
        "--ci-bootstrap-samples",
        type=int,
        default=0,
        help="Episode bootstrap resamples per dataset for interval estimation.",
    )
    parser.add_argument(
        "--interval-mode",
        type=str,
        default="none",
        choices=["none", "bootstrap", "analytic", "both"],
        help="Which estimator interval columns to compute during the sweep.",
    )
    parser.add_argument(
        "--estimators",
        type=str,
        default="is_pdis,dr_oracle,dm_tabular,fqe_linear",
        help="Comma-separated estimator keys used for focused summaries and interval work.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override multiprocessing worker count. Use -1 to auto-use all detected CPUs.",
    )
    parser.add_argument(
        "--mp-chunksize",
        type=int,
        default=None,
        help="Override multiprocessing chunk size for the sweep worker pool.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = SweepConfig.from_dict(load_yaml(args.config))
    config.ci_level = float(args.ci_level)
    config.ci_bootstrap_samples = int(args.ci_bootstrap_samples)
    config.interval_mode = str(args.interval_mode)
    config.analysis_estimators = [part.strip() for part in args.estimators.split(",") if part.strip()]
    if args.num_workers is not None:
        config.num_workers = int(args.num_workers)
    if args.mp_chunksize is not None:
        config.mp_chunksize = int(args.mp_chunksize)
    df, run_dir, result_path = run_sweep(config)

    fig_dir = run_dir / args.fig_subdir
    interval_methods = [m for m in [args.interval_mode] if m not in {"none", "both"}]
    if args.interval_mode == "both":
        interval_methods = ["analytic", "bootstrap"]
    artifacts = generate_run_artifacts(
        df=df,
        output_dir=fig_dir,
        estimator_keys=config.analysis_estimators,
        fixed_alpha=args.fixed_alpha,
        fixed_beta=args.fixed_beta,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_max_points=args.bootstrap_max_points,
        ci_level=args.ci_level,
        interval_methods=interval_methods or None,
        paper_mode=args.paper_mode,
    )

    print(f"Rows: {len(df)}")
    print(f"Run dir: {run_dir}")
    print(f"Results: {result_path}")
    print(f"Figures: {fig_dir}")
    print(f"Report: {fig_dir / 'benchmark_report.csv'}")
    print(f"Estimator Summary: {fig_dir / 'estimator_summary.csv'}")
    print(f"Condition Summary: {fig_dir / 'condition_summary.csv'}")
    print(f"Bias/Variance Summary: {fig_dir / 'bias_variance_summary.csv'}")
    print(f"CI Interval Summary: {fig_dir / 'ci_interval_summary.csv'}")
    print(f"CI Coverage Summary: {fig_dir / 'ci_coverage_summary.csv'}")
    print(f"Diagnostic Comparability: {fig_dir / 'diagnostic_comparability.csv'}")
    if (fig_dir / "chain_bandit_sensitivity_summary.csv").exists():
        print(f"Chain Bandit Sensitivity Summary: {fig_dir / 'chain_bandit_sensitivity_summary.csv'}")
    print(f"Trial Scorecard: {fig_dir / 'trial_scorecard.csv'}")
    if args.paper_mode:
        print(f"Paper Claims: {fig_dir / 'paper_claims.csv'}")
        print(f"Paper Claim Summary: {fig_dir / 'paper_claim_summary.csv'}")
    report = artifacts["benchmark_report"]
    if not report.empty:
        print(report.to_string(index=False))
    if args.paper_mode and not artifacts["paper_claims"].empty:
        print("\nPaper Claims:")
        print(artifacts["paper_claims"].to_string(index=False))


if __name__ == "__main__":
    main()
