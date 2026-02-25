#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ess_ope.evaluation.summary import (
    PaperClaimConfig,
    SummaryConfig,
    build_condition_summary,
    build_estimator_summary,
    build_paper_claim_summary,
    build_paper_claims_table,
)
from ess_ope.plotting.benchmark_figures import generate_benchmark_figures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate benchmark plots/report from an existing sweep results file")
    parser.add_argument(
        "--results",
        type=str,
        default="results/latest/sweep_results.parquet",
        help="Path to sweep_results parquet/csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/latest/figures",
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
    return parser.parse_args()


def _load_results(path: Path) -> pd.DataFrame:
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
    stage_bar = tqdm(total=7, desc="Interpret results", disable=not args.progress)
    df = _load_results(Path(args.results))
    stage_bar.update(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_bar.update(1)

    report = generate_benchmark_figures(
        df=df,
        output_dir=output_dir,
        fixed_alpha=args.fixed_alpha,
        fixed_beta=args.fixed_beta,
    )
    stage_bar.update(1)
    summary_cfg = SummaryConfig(
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_max_points=args.bootstrap_max_points,
        show_progress=args.progress,
    )
    estimator_summary = build_estimator_summary(df, config=summary_cfg)
    stage_bar.update(1)
    condition_summary = build_condition_summary(df, show_progress=args.progress)
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
    paper_claims_path = output_dir / "paper_claims.csv"
    paper_claim_summary_path = output_dir / "paper_claim_summary.csv"
    report.to_csv(report_path, index=False)
    estimator_summary.to_csv(estimator_summary_path, index=False)
    condition_summary.to_csv(condition_summary_path, index=False)
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
