#!/usr/bin/env python3
from __future__ import annotations

import argparse

from ess_ope.study import StudyConfig, run_study


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full ESS-vs-uncertainty paper study.")
    parser.add_argument("--config", required=True, help="Path to study YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = StudyConfig.from_yaml(args.config)
    df, run_dir, _ = run_study(cfg)
    print(f"Rows: {len(df)}")
    print(f"Run dir: {run_dir}")
    print(f"Replicate results: {run_dir / cfg.output_subdir / 'replicate_results.csv'}")
    print(f"Artifacts: {run_dir / cfg.output_subdir}")


if __name__ == "__main__":
    main()
