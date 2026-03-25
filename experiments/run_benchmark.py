#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from ess_ope.evaluation.sweep import SweepConfig, run_sweep
from ess_ope.evaluation.summary import (
    PaperClaimConfig,
    SummaryConfig,
    build_bias_variance_summary,
    build_ci_coverage_summary,
    build_ci_interval_summary,
    build_condition_summary,
    build_diagnostic_comparability_summary,
    build_estimator_summary,
    build_paper_claim_summary,
    build_paper_claims_table,
)
from ess_ope.plotting.benchmark_figures import generate_benchmark_figures
from ess_ope.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ESS benchmark sweep and generate comparison figures for IS/DR/DM/FQE"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/sweeps/random_mdp_ultra.yaml",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = SweepConfig.from_dict(load_yaml(args.config))
    config.ci_level = float(args.ci_level)
    config.ci_bootstrap_samples = int(args.ci_bootstrap_samples)
    config.interval_mode = str(args.interval_mode)
    config.analysis_estimators = [part.strip() for part in args.estimators.split(",") if part.strip()]
    df, run_dir, result_path = run_sweep(config)

    fig_dir = run_dir / args.fig_subdir
    fig_dir.mkdir(parents=True, exist_ok=True)
    summary_cfg = SummaryConfig(
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_max_points=args.bootstrap_max_points,
    )
    estimator_summary = build_estimator_summary(df, estimator_keys=config.analysis_estimators, config=summary_cfg)
    condition_summary = build_condition_summary(df)
    bias_variance_summary = build_bias_variance_summary(df, estimator_keys=config.analysis_estimators)
    interval_methods = [m for m in [args.interval_mode] if m not in {"none", "both"}]
    if args.interval_mode == "both":
        interval_methods = ["analytic", "bootstrap"]
    ci_interval_summary = build_ci_interval_summary(df, estimator_keys=config.analysis_estimators, methods=interval_methods or None)
    ci_coverage_summary = build_ci_coverage_summary(ci_interval_summary, ci_level=args.ci_level)
    diagnostic_comparability = build_diagnostic_comparability_summary(
        df,
        ci_coverage_summary=ci_coverage_summary,
        estimator_keys=config.analysis_estimators,
    )
    report = generate_benchmark_figures(
        df=df,
        output_dir=fig_dir,
        fixed_alpha=args.fixed_alpha,
        fixed_beta=args.fixed_beta,
        bias_variance_summary=bias_variance_summary,
        ci_interval_summary=ci_interval_summary,
        ci_coverage_summary=ci_coverage_summary,
    )
    paper_claims = build_paper_claims_table(
        df=df,
        estimator_summary=estimator_summary,
        benchmark_report=report,
        config=PaperClaimConfig(),
    )
    paper_claim_summary = build_paper_claim_summary(paper_claims)

    report_path = fig_dir / "benchmark_report.csv"
    estimator_summary_path = fig_dir / "estimator_summary.csv"
    condition_summary_path = fig_dir / "condition_summary.csv"
    bias_variance_summary_path = fig_dir / "bias_variance_summary.csv"
    ci_interval_summary_path = fig_dir / "ci_interval_summary.csv"
    ci_coverage_summary_path = fig_dir / "ci_coverage_summary.csv"
    diagnostic_comparability_path = fig_dir / "diagnostic_comparability.csv"
    paper_claims_path = fig_dir / "paper_claims.csv"
    paper_claim_summary_path = fig_dir / "paper_claim_summary.csv"
    report.to_csv(report_path, index=False)
    estimator_summary.to_csv(estimator_summary_path, index=False)
    condition_summary.to_csv(condition_summary_path, index=False)
    bias_variance_summary.to_csv(bias_variance_summary_path, index=False)
    ci_interval_summary.to_csv(ci_interval_summary_path, index=False)
    ci_coverage_summary.to_csv(ci_coverage_summary_path, index=False)
    diagnostic_comparability.to_csv(diagnostic_comparability_path, index=False)
    if args.paper_mode:
        paper_claims.to_csv(paper_claims_path, index=False)
        paper_claim_summary.to_csv(paper_claim_summary_path, index=False)

    print(f"Rows: {len(df)}")
    print(f"Run dir: {run_dir}")
    print(f"Results: {result_path}")
    print(f"Figures: {fig_dir}")
    print(f"Report: {report_path}")
    print(f"Estimator Summary: {estimator_summary_path}")
    print(f"Condition Summary: {condition_summary_path}")
    print(f"Bias/Variance Summary: {bias_variance_summary_path}")
    print(f"CI Interval Summary: {ci_interval_summary_path}")
    print(f"CI Coverage Summary: {ci_coverage_summary_path}")
    print(f"Diagnostic Comparability: {diagnostic_comparability_path}")
    if args.paper_mode:
        print(f"Paper Claims: {paper_claims_path}")
        print(f"Paper Claim Summary: {paper_claim_summary_path}")
    if not report.empty:
        print(report.to_string(index=False))
    if args.paper_mode and not paper_claims.empty:
        print("\nPaper Claims:")
        print(paper_claims.to_string(index=False))


if __name__ == "__main__":
    main()
