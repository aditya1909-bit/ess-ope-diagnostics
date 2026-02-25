#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from ess_ope.evaluation.sweep import SweepConfig, run_sweep
from ess_ope.evaluation.summary import (
    PaperClaimConfig,
    SummaryConfig,
    build_condition_summary,
    build_estimator_summary,
    build_paper_claim_summary,
    build_paper_claims_table,
)
from ess_ope.plotting.benchmark_figures import generate_benchmark_figures
from ess_ope.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ESS benchmark sweep and generate comparison figures for IS vs DM/FQE/MRDR"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/sweeps/random_mdp_robust.yaml",
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
        default=400,
        help="Bootstrap samples for correlation confidence intervals in summary tables",
    )
    parser.add_argument(
        "--bootstrap-max-points",
        type=int,
        default=5000,
        help="Max points used for bootstrap correlation CI computation",
    )
    parser.add_argument(
        "--paper-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write CI-aware paper claims tables (paper_claims.csv, paper_claim_summary.csv).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = SweepConfig.from_dict(load_yaml(args.config))
    df, run_dir, result_path = run_sweep(config)

    fig_dir = run_dir / args.fig_subdir
    fig_dir.mkdir(parents=True, exist_ok=True)
    report = generate_benchmark_figures(
        df=df,
        output_dir=fig_dir,
        fixed_alpha=args.fixed_alpha,
        fixed_beta=args.fixed_beta,
    )
    summary_cfg = SummaryConfig(
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_max_points=args.bootstrap_max_points,
    )
    estimator_summary = build_estimator_summary(df, config=summary_cfg)
    condition_summary = build_condition_summary(df)
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
    paper_claims_path = fig_dir / "paper_claims.csv"
    paper_claim_summary_path = fig_dir / "paper_claim_summary.csv"
    report.to_csv(report_path, index=False)
    estimator_summary.to_csv(estimator_summary_path, index=False)
    condition_summary.to_csv(condition_summary_path, index=False)
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
