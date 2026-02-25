from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from ess_ope.utils.config import dump_yaml


def git_commit_hash() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return out
    except Exception:
        return "unknown"


def timestamp_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def create_run_dir(results_root: str | Path, run_name: str) -> Path:
    root = Path(results_root)
    run_dir = root / f"{timestamp_utc()}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_run_metadata(run_dir: Path, config: Dict[str, Any]) -> None:
    config_path = run_dir / "config.yaml"
    dump_yaml(config_path, config)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit_hash(),
    }
    with (run_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def save_results_table(df: pd.DataFrame, run_dir: Path, stem: str) -> Path:
    parquet_path = run_dir / f"{stem}.parquet"
    csv_path = run_dir / f"{stem}.csv"

    try:
        df.to_parquet(parquet_path, index=False)
        selected_path = parquet_path
    except Exception:
        df.to_csv(csv_path, index=False)
        selected_path = csv_path

    df.to_csv(csv_path, index=False)
    return selected_path


def update_latest_pointer(results_root: str | Path, run_dir: Path) -> Optional[Path]:
    root = Path(results_root)
    latest = root / "latest"
    try:
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(run_dir.name)
        return latest
    except Exception:
        return None
