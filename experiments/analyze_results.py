#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ess_ope.evaluation.reporting import generate_run_artifacts
from ess_ope.utils.logging import refresh_latest_pointer, resolve_latest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate benchmark plots/report from an existing sweep results file")
    parser.add_argument(
        "--results",
        type=str,
        default="results/latest_random_mdp/sweep_results.csv",
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
    artifacts = generate_run_artifacts(
        df=df,
        output_dir=output_dir,
        estimator_keys=estimator_keys,
        fixed_alpha=args.fixed_alpha,
        fixed_beta=args.fixed_beta,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_max_points=args.bootstrap_max_points,
        ci_level=args.ci_level,
        interval_methods=interval_methods or None,
        paper_mode=args.paper_mode,
        progress=args.progress,
    )
    stage_bar.update(10)
    stage_bar.close()

    print(f"Loaded rows: {len(df)}")
    print(f"Output dir: {output_dir}")
    print(f"Report: {output_dir / 'benchmark_report.csv'}")
    print(f"Estimator Summary: {output_dir / 'estimator_summary.csv'}")
    print(f"Condition Summary: {output_dir / 'condition_summary.csv'}")
    print(f"Bias/Variance Summary: {output_dir / 'bias_variance_summary.csv'}")
    print(f"CI Interval Summary: {output_dir / 'ci_interval_summary.csv'}")
    print(f"CI Coverage Summary: {output_dir / 'ci_coverage_summary.csv'}")
    print(f"Diagnostic Comparability: {output_dir / 'diagnostic_comparability.csv'}")
    if (output_dir / "chain_bandit_sensitivity_summary.csv").exists():
        print(f"Chain Bandit Sensitivity Summary: {output_dir / 'chain_bandit_sensitivity_summary.csv'}")
    print(f"Trial Scorecard: {output_dir / 'trial_scorecard.csv'}")
    if args.paper_mode:
        print(f"Paper Claims: {output_dir / 'paper_claims.csv'}")
        print(f"Paper Claim Summary: {output_dir / 'paper_claim_summary.csv'}")
    if not artifacts["benchmark_report"].empty:
        print(artifacts["benchmark_report"].to_string(index=False))
    if args.paper_mode and not artifacts["paper_claims"].empty:
        print("\nPaper Claims:")
        print(artifacts["paper_claims"].to_string(index=False))


if __name__ == "__main__":
    main()
