#!/usr/bin/env python3
from __future__ import annotations

import argparse

from ess_ope.study import analyze_saved_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze saved ESS-vs-uncertainty replicate results.")
    parser.add_argument("--results", required=True, help="Path to replicate_results.csv or .parquet.")
    parser.add_argument("--output-dir", default=None, help="Optional artifact output directory.")
    parser.add_argument("--primary-level", type=float, default=0.9, help="Primary CI level for width/coverage proxying.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyze_saved_results(results_path=args.results, output_dir=args.output_dir, primary_level=float(args.primary_level))


if __name__ == "__main__":
    main()
