#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ess_ope.v2.runner import analyze_phase_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze existing v2 replicate results into tables and figures.")
    parser.add_argument("--results", type=str, required=True, help="Path to replicate_results.csv/parquet.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for analysis outputs.")
    return parser.parse_args()


def _load(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)


def main() -> None:
    args = parse_args()
    df = _load(args.results)
    artifacts = analyze_phase_results(df, args.output_dir)
    print(f"Rows: {len(df)}")
    print(f"Output dir: {Path(args.output_dir)}")
    print(f"Table A: {Path(args.output_dir) / 'table_a_estimator_summary.csv'}")
    print(f"Table B: {Path(args.output_dir) / 'table_b_diagnostic_quality.csv'}")
    print(f"Figures written: {len(list(Path(args.output_dir).glob('v2_fig*.png')))}")
    if not artifacts["table_a_estimator_summary"].empty:
        print(artifacts["table_a_estimator_summary"].head().to_string(index=False))


if __name__ == "__main__":
    main()
