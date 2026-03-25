from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from ess_ope.utils.config import dump_yaml

if TYPE_CHECKING:
    import pandas as pd


_RUN_DIR_RE = re.compile(r"^\d{8}_\d{6}_.+")
_LATEST_ALIAS_RE = re.compile(r"^latest(?:_[A-Za-z0-9_]+)?$")
_LATEST_MANIFEST = "latest_runs.json"


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


def save_results_table(df: "pd.DataFrame", run_dir: Path, stem: str) -> Path:
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


def latest_run_dir(results_root: str | Path) -> Optional[Path]:
    root = Path(results_root)
    if not root.exists():
        return None

    candidates = [
        path
        for path in root.iterdir()
        if path.is_dir() and _RUN_DIR_RE.match(path.name)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.name)


def _latest_manifest_path(results_root: str | Path) -> Path:
    return Path(results_root) / _LATEST_MANIFEST


def load_latest_manifest(results_root: str | Path) -> Dict[str, str]:
    manifest_path = _latest_manifest_path(results_root)
    if not manifest_path.exists():
        return {}
    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    manifest: Dict[str, str] = {}
    for key, value in payload.items():
        if isinstance(key, str) and isinstance(value, str) and _LATEST_ALIAS_RE.match(key):
            manifest[key] = value
    return manifest


def save_latest_manifest(results_root: str | Path, manifest: Dict[str, str]) -> Path:
    manifest_path = _latest_manifest_path(results_root)
    ordered = {key: manifest[key] for key in sorted(manifest)}
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(ordered, f, indent=2, sort_keys=True)
        f.write("\n")
    return manifest_path


def latest_run_dir_for_env(results_root: str | Path, env_name: str) -> Optional[Path]:
    root = Path(results_root)
    if not root.exists():
        return None

    token = f"_{env_name}_"
    suffix = f"_{env_name}"
    candidates = [
        path
        for path in root.iterdir()
        if path.is_dir()
        and _RUN_DIR_RE.match(path.name)
        and (token in path.name or path.name.endswith(suffix))
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.name)


def resolve_latest_path(path: str | Path) -> Path:
    p = Path(path)
    parts = list(p.parts)

    idx = next((i for i, part in enumerate(parts) if _LATEST_ALIAS_RE.match(part)), None)
    if idx is None:
        return p

    if idx == 0:
        results_root = Path("results")
        suffix = Path(*parts[1:]) if len(parts) > 1 else Path()
    else:
        results_root = Path(*parts[:idx])
        suffix = Path(*parts[idx + 1 :]) if idx + 1 < len(parts) else Path()

    alias = parts[idx]
    manifest = load_latest_manifest(results_root)
    manifest_target = manifest.get(alias)
    latest_dir = (results_root / manifest_target) if manifest_target else None
    if latest_dir is not None and not latest_dir.exists():
        latest_dir = None
    if latest_dir is None:
        if alias == "latest":
            latest_dir = latest_run_dir(results_root)
        else:
            env_name = alias.replace("latest_", "", 1)
            latest_dir = latest_run_dir_for_env(results_root, env_name)
    if latest_dir is None:
        return p
    return latest_dir / suffix


def refresh_latest_pointer(results_root: str | Path) -> Optional[Path]:
    root = Path(results_root)
    latest_dir = latest_run_dir(root)
    if latest_dir is None:
        return None
    manifest: Dict[str, str] = {"latest": latest_dir.name}
    update_latest_pointer(root, latest_dir)
    for env_name in ["random_mdp", "chain_bandit"]:
        env_latest = latest_run_dir_for_env(root, env_name)
        if env_latest is not None:
            alias = f"latest_{env_name}"
            update_latest_pointer(root, env_latest, alias=alias)
            manifest[alias] = env_latest.name
    save_latest_manifest(root, manifest)
    return root / "latest"


def update_latest_pointer(results_root: str | Path, run_dir: Path, alias: str = "latest") -> Optional[Path]:
    root = Path(results_root)
    latest = root / alias
    try:
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(run_dir.name)
        return latest
    except Exception:
        return None
