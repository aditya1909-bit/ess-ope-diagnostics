#!/usr/bin/env python3
from __future__ import annotations

import argparse

from ess_ope.study import ExperimentConfig, run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one ESS-vs-uncertainty paper experiment.")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)
    df, run_dir, _ = run_experiment(cfg)
    print(f"Rows: {len(df)}")
    print(f"Run dir: {run_dir}")
    print(f"Replicate results: {run_dir / cfg.output_subdir / 'replicate_results.csv'}")
    print(f"Artifacts: {run_dir / cfg.output_subdir}")


if __name__ == "__main__":
    main()
