#!/usr/bin/env python3
from __future__ import annotations

import argparse

from ess_ope.v2.config import SuiteConfig
from ess_ope.v2.runner import run_suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full v2 OPE uncertainty-diagnostics suite.")
    parser.add_argument("--config", type=str, required=True, help="Path to suite YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SuiteConfig.from_yaml(args.config)
    df, run_dir = run_suite(cfg)
    print(f"Rows: {len(df)}")
    print(f"Run dir: {run_dir}")
    print(f"Replicate results: {run_dir / 'replicate_results.csv'}")
    print(f"Artifacts: {run_dir / cfg.output_subdir}")


if __name__ == "__main__":
    main()
