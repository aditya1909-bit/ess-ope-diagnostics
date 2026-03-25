#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ess_ope.evaluation.summary import (
    PaperClaimConfig,
    SummaryConfig,
    build_bias_variance_summary,
    build_chain_bandit_sensitivity_summary,
    build_ci_coverage_summary,
    build_ci_interval_summary,
    build_condition_summary,
    build_diagnostic_comparability_summary,
    build_estimator_summary,
    build_paper_claim_summary,
    build_paper_claims_table,
)
from ess_ope.plotting.benchmark_figures import generate_benchmark_figures
from ess_ope.utils.logging import refresh_latest_pointer, resolve_latest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate benchmark plots/report from an existing sweep results file")
    parser.add_argument(
        "--results",
        type=str,
        default="results/latest_random_mdp/sweep_results.parquet",
        help="Path to sweep_results parquet/csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/latest_random_mdp/figures",
        help="Directory for generated figures/report",
    )
    parser.add_argument("--fixed-alpha", type=float, default=None)
    parser.add_argument("--fixed-beta", type=float, default=0.0)
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
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show progress bars for interpretation pipeline stages and summaries.",
    )
    parser.add_argument(
        "--ci-level",
        type=float,
        default=0.95,
        help="Confidence level used when aggregating interval summaries.",
    )
    parser.add_argument(
        "--interval-mode",
        type=str,
        default="both",
        choices=["none", "bootstrap", "analytic", "both"],
        help="Which interval methods to summarize from saved sweep columns.",
    )
    parser.add_argument(
        "--estimators",
        type=str,
        default="is_pdis,dr_oracle,dm_tabular,fqe_linear",
        help="Comma-separated estimator keys used for focused summaries.",
    )
    return parser.parse_args()


def _load_results(path: Path) -> pd.DataFrame:
    path = resolve_latest_path(path)
    if path.exists():
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        if path.suffix == ".csv":
            return pd.read_csv(path)

    parquet = path.with_suffix(".parquet")
    csv = path.with_suffix(".csv")
    if parquet.exists():
        return pd.read_parquet(parquet)
    if csv.exists():
        return pd.read_csv(csv)

    raise FileNotFoundError(f"No results found for: {path}")


def main() -> None:
    args = parse_args()
    refresh_latest_pointer("results")
    stage_bar = tqdm(total=12, desc="Interpret results", disable=not args.progress)
    df = _load_results(Path(args.results))
    stage_bar.update(1)

    output_dir = resolve_latest_path(Path(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_bar.update(1)

    estimator_keys = [part.strip() for part in args.estimators.split(",") if part.strip()]
    interval_methods = [m for m in [args.interval_mode] if m not in {"none", "both"}]
    if args.interval_mode == "both":
        interval_methods = ["analytic", "bootstrap"]
    summary_cfg = SummaryConfig(
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_max_points=args.bootstrap_max_points,
        show_progress=args.progress,
    )
    estimator_summary = build_estimator_summary(df, estimator_keys=estimator_keys, config=summary_cfg)
    stage_bar.update(1)
    condition_summary = build_condition_summary(df, show_progress=args.progress)
    stage_bar.update(1)
    bias_variance_summary = build_bias_variance_summary(df, estimator_keys=estimator_keys)
    stage_bar.update(1)
    ci_interval_summary = build_ci_interval_summary(df, estimator_keys=estimator_keys, methods=interval_methods or None)
    stage_bar.update(1)
    ci_coverage_summary = build_ci_coverage_summary(ci_interval_summary, ci_level=args.ci_level)
    stage_bar.update(1)
    diagnostic_comparability = build_diagnostic_comparability_summary(
        df,
        ci_coverage_summary=ci_coverage_summary,
        estimator_keys=estimator_keys,
    )
    stage_bar.update(1)
    chain_bandit_sensitivity = build_chain_bandit_sensitivity_summary(df)
    stage_bar.update(1)
    report = generate_benchmark_figures(
        df=df,
        output_dir=output_dir,
        fixed_alpha=args.fixed_alpha,
        fixed_beta=args.fixed_beta,
        bias_variance_summary=bias_variance_summary,
        ci_interval_summary=ci_interval_summary,
        ci_coverage_summary=ci_coverage_summary,
        estimator_summary=estimator_summary,
        diagnostic_comparability=diagnostic_comparability,
    )
    stage_bar.update(1)
    paper_claims = build_paper_claims_table(
        df=df,
        estimator_summary=estimator_summary,
        benchmark_report=report,
        config=PaperClaimConfig(),
    )
    paper_claim_summary = build_paper_claim_summary(paper_claims)
    stage_bar.update(1)
    report_path = output_dir / "benchmark_report.csv"
    estimator_summary_path = output_dir / "estimator_summary.csv"
    condition_summary_path = output_dir / "condition_summary.csv"
    bias_variance_summary_path = output_dir / "bias_variance_summary.csv"
    ci_interval_summary_path = output_dir / "ci_interval_summary.csv"
    ci_coverage_summary_path = output_dir / "ci_coverage_summary.csv"
    diagnostic_comparability_path = output_dir / "diagnostic_comparability.csv"
    chain_bandit_sensitivity_path = output_dir / "chain_bandit_sensitivity_summary.csv"
    paper_claims_path = output_dir / "paper_claims.csv"
    paper_claim_summary_path = output_dir / "paper_claim_summary.csv"
    report.to_csv(report_path, index=False)
    estimator_summary.to_csv(estimator_summary_path, index=False)
    condition_summary.to_csv(condition_summary_path, index=False)
    bias_variance_summary.to_csv(bias_variance_summary_path, index=False)
    ci_interval_summary.to_csv(ci_interval_summary_path, index=False)
    ci_coverage_summary.to_csv(ci_coverage_summary_path, index=False)
    diagnostic_comparability.to_csv(diagnostic_comparability_path, index=False)
    if not chain_bandit_sensitivity.empty:
        chain_bandit_sensitivity.to_csv(chain_bandit_sensitivity_path, index=False)
    if args.paper_mode:
        paper_claims.to_csv(paper_claims_path, index=False)
        paper_claim_summary.to_csv(paper_claim_summary_path, index=False)
    stage_bar.update(1)
    stage_bar.close()

    print(f"Loaded rows: {len(df)}")
    print(f"Output dir: {output_dir}")
    print(f"Report: {report_path}")
    print(f"Estimator Summary: {estimator_summary_path}")
    print(f"Condition Summary: {condition_summary_path}")
    print(f"Bias/Variance Summary: {bias_variance_summary_path}")
    print(f"CI Interval Summary: {ci_interval_summary_path}")
    print(f"CI Coverage Summary: {ci_coverage_summary_path}")
    print(f"Diagnostic Comparability: {diagnostic_comparability_path}")
    if not chain_bandit_sensitivity.empty:
        print(f"Chain Bandit Sensitivity Summary: {chain_bandit_sensitivity_path}")
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
