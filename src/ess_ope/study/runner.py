from __future__ import annotations

import hashlib
import json
import os
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from ess_ope.data.generate import generate_offline_dataset
from ess_ope.evaluation.ground_truth import dynamic_programming_value
from ess_ope.study.analysis import generate_study_artifacts
from ess_ope.study.config import ExperimentConfig, StudyConfig, merge_experiment_overrides
from ess_ope.study.envs import build_environment_bundle
from ess_ope.study.estimators import DISPLAY_NAMES, evaluate_estimator
from ess_ope.study.inference import episode_bootstrap, summarize_intervals
from ess_ope.utils.logging import create_run_dir, refresh_latest_pointer, resolve_latest_path, save_run_metadata


ETA_BAR_FORMAT = "{desc:<22} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"


def _seed_hash(*values: Any) -> int:
    text = "|".join(map(str, values)).encode("utf-8")
    digest = hashlib.blake2b(text, digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32 - 1)


def _config_hash(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _iter_conditions(grid: Dict[str, Iterable[Any]]) -> Iterable[Dict[str, Any]]:
    keys = list(grid)
    if not keys:
        yield {}
        return
    def _rec(idx: int, current: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
        if idx == len(keys):
            yield dict(current)
            return
        key = keys[idx]
        values = list(grid.get(key, []))
        if not values:
            current[key] = None
            yield from _rec(idx + 1, current)
            current.pop(key, None)
            return
        for value in values:
            current[key] = value
            yield from _rec(idx + 1, current)
        current.pop(key, None)
    yield from _rec(0, {})


def _resolve_num_workers(config: ExperimentConfig) -> int:
    requested = int(config.num_workers)
    if requested > 0:
        return requested
    detected = os.cpu_count() or 1
    return max(1, detected)


def _build_tasks(config: ExperimentConfig) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    config_dict = config.to_dict()
    for seed in config.seeds:
        for condition in _iter_conditions(config.grid):
            sample_size = int(condition.get("sample_size", config.environment.get("sample_size", 100)))
            for replicate_id in range(int(config.replicates)):
                tasks.append(
                    {
                        "config": config_dict,
                        "seed": int(seed),
                        "condition": condition,
                        "replicate_id": int(replicate_id),
                        "sample_size": sample_size,
                    }
                )
    return tasks


def _condition_id(experiment_id: str, condition: Dict[str, Any]) -> str:
    parts = [experiment_id] + [f"{key}={condition[key]}" for key in sorted(condition)]
    return "|".join(parts)


def _base_point_row(
    config: ExperimentConfig,
    bundle,
    condition: Dict[str, Any],
    replicate_id: int,
    dataset_seed: int,
    truth_value: float,
    estimator_result,
) -> Dict[str, Any]:
    row = {
        "experiment_id": config.experiment_id,
        "experiment_title": config.title,
        "env_name": bundle.metadata["env_name"],
        "condition_id": _condition_id(config.experiment_id, condition),
        "replicate_id": int(replicate_id),
        "dataset_seed": int(dataset_seed),
        "estimator_key": estimator_result.estimator_key,
        "estimator_label": DISPLAY_NAMES[estimator_result.estimator_key],
        "estimator_family": estimator_result.estimator_family,
        "ci_method": "point",
        "ci_level": np.nan,
        "estimate": float(estimator_result.estimate),
        "true_value": float(truth_value),
        "error": float(estimator_result.estimate - truth_value),
        "abs_error": float(abs(estimator_result.estimate - truth_value)),
        "squared_error": float((estimator_result.estimate - truth_value) ** 2),
        "shared_wess": float(estimator_result.shared_wess),
        "wess_native_applicable": bool(estimator_result.wess_native_applicable),
        "ci_low": np.nan,
        "ci_high": np.nan,
        "ci_width": np.nan,
        "covered": np.nan,
        "bootstrap_variance": np.nan,
        "bootstrap_runtime_sec": 0.0,
        "estimator_runtime_sec": float(estimator_result.runtime_sec),
        "truth_method": config.truth_method,
    }
    row.update(condition)
    return row


def _append_interval_rows(
    rows: List[Dict[str, Any]],
    base_row: Dict[str, Any],
    intervals: Dict[str, Any],
    bootstrap_variance: float,
    bootstrap_runtime_sec: float,
    truth_value: float,
) -> None:
    for method, interval in intervals.items():
        row = dict(base_row)
        row["ci_method"] = method
        row["ci_low"] = float(interval.low)
        row["ci_high"] = float(interval.high)
        row["ci_width"] = float(interval.width)
        row["covered"] = float(interval.low <= truth_value <= interval.high)
        row["bootstrap_variance"] = float(bootstrap_variance)
        row["bootstrap_runtime_sec"] = float(bootstrap_runtime_sec)
        rows.append(row)


def _evaluate_replicate(
    config: ExperimentConfig,
    seed: int,
    condition: Dict[str, Any],
    replicate_id: int,
    sample_size: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    bundle = build_environment_bundle(config.environment, seed=seed, condition=condition)
    truth = dynamic_programming_value(bundle.env, bundle.target_policy, gamma=1.0)
    methods = list(config.intervals.get("methods", []))
    if bool(config.intervals.get("enable_concentration", False)):
        methods.append("concentration_empirical_bernstein")
    dataset_seed = _seed_hash(config.experiment_id, seed, replicate_id, sample_size, json.dumps(condition, sort_keys=True))
    dataset = generate_offline_dataset(
        env=bundle.env,
        behavior_policy=bundle.behavior_policy,
        num_episodes=sample_size,
        horizon=bundle.env.horizon,
        seed=dataset_seed,
    )
    for estimator_key in config.estimators:
        estimator_result = evaluate_estimator(
            estimator_key=estimator_key,
            dataset=dataset,
            target_policy=bundle.target_policy,
            behavior_policy=bundle.behavior_policy,
            env=bundle.env,
            gamma=1.0,
        )
        base_row = _base_point_row(config, bundle, condition | {"seed": seed, **bundle.metadata}, replicate_id, dataset_seed, truth.value, estimator_result)
        rows.append(base_row)

        bootstrap = episode_bootstrap(
            dataset=dataset,
            estimate_fn=lambda ds, key=estimator_key: evaluate_estimator(
                estimator_key=key,
                dataset=ds,
                target_policy=bundle.target_policy,
                behavior_policy=bundle.behavior_policy,
                env=bundle.env,
                gamma=1.0,
            ).estimate,
            n_boot=int(config.intervals.get("bootstrap_samples", 0)),
            seed=_seed_hash(dataset_seed, estimator_key, "bootstrap"),
            subsample_ratio=float(config.intervals.get("subsample_ratio", 1.0)),
        )
        for ci_level in config.intervals.get("levels", [0.9, 0.95]):
            point_row = dict(base_row)
            point_row["ci_level"] = float(ci_level)
            intervals = summarize_intervals(
                point_estimate=estimator_result.estimate,
                bootstrap=bootstrap,
                contributions=estimator_result.episode_contributions,
                ci_level=float(ci_level),
                methods=methods,
            )
            _append_interval_rows(
                rows=rows,
                base_row=point_row,
                intervals=intervals,
                bootstrap_variance=bootstrap.variance,
                bootstrap_runtime_sec=bootstrap.runtime_sec,
                truth_value=truth.value,
            )
    return rows


def _evaluate_replicate_task(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    cfg = ExperimentConfig.from_dict(task["config"])
    return _evaluate_replicate(
        cfg,
        seed=int(task["seed"]),
        condition=dict(task["condition"]),
        replicate_id=int(task["replicate_id"]),
        sample_size=int(task["sample_size"]),
    )


def _finalize_run(
    raw_df: pd.DataFrame,
    run_dir: Path,
    config_payload: Dict[str, Any],
    output_subdir: str,
    primary_level: float,
    elapsed_sec: float,
) -> Dict[str, pd.DataFrame]:
    output_dir = run_dir / output_subdir
    artifacts = generate_study_artifacts(raw_df, output_dir=output_dir, primary_level=primary_level)
    from ess_ope.study.plotting import generate_study_figures

    generate_study_figures(
        raw_df=raw_df,
        point_df=artifacts["point_estimates"],
        table_2=artifacts["table_2_diagnostic_usefulness"],
        output_dir=output_dir,
        primary_level=primary_level,
    )
    save_run_metadata(run_dir, config_payload)
    metadata_path = run_dir / "metadata.json"
    with metadata_path.open("r+", encoding="utf-8") as handle:
        metadata = json.load(handle)
        metadata.update(
            {
                "config_hash": _config_hash(config_payload),
                "elapsed_sec": float(elapsed_sec),
                "truth_generation_method": "dynamic_programming",
            }
        )
        handle.seek(0)
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
        handle.truncate()
    refresh_latest_pointer(run_dir.parent)
    return artifacts


def run_experiment(config: ExperimentConfig) -> Tuple[pd.DataFrame, Path, Dict[str, pd.DataFrame]]:
    start = perf_counter()
    run_dir = create_run_dir(config.results_root, config.experiment_id)
    rows: List[Dict[str, Any]] = []
    tasks = _build_tasks(config)
    num_workers = _resolve_num_workers(config)
    pbar = tqdm(
        total=len(tasks),
        desc=f"Experiment {config.experiment_id}",
        dynamic_ncols=True,
        bar_format=ETA_BAR_FORMAT,
        smoothing=0.1,
    )
    if num_workers <= 1:
        for task in tasks:
            rows.extend(_evaluate_replicate_task(task))
            pbar.update(1)
    else:
        ctx = get_context("spawn")
        with ctx.Pool(processes=num_workers) as pool:
            for result_rows in pool.imap_unordered(_evaluate_replicate_task, tasks, chunksize=max(1, int(config.mp_chunksize))):
                rows.extend(result_rows)
                pbar.update(1)
    pbar.close()
    raw_df = pd.DataFrame(rows)
    artifacts = _finalize_run(
        raw_df=raw_df,
        run_dir=run_dir,
        config_payload=config.to_dict(),
        output_subdir=config.output_subdir,
        primary_level=float(config.intervals.get("primary_level", 0.9)),
        elapsed_sec=perf_counter() - start,
    )
    return raw_df, run_dir, artifacts


def run_study(config: StudyConfig) -> Tuple[pd.DataFrame, Path, Dict[str, pd.DataFrame]]:
    start = perf_counter()
    run_dir = create_run_dir(config.results_root, config.name)
    experiment_frames: List[pd.DataFrame] = []
    experiment_payloads: List[Dict[str, Any]] = []
    experiment_cfgs = [merge_experiment_overrides(ExperimentConfig.from_yaml(path), config.overrides) for path in config.experiment_configs]
    total_tasks = sum(len(_build_tasks(experiment_cfg)) for experiment_cfg in experiment_cfgs)
    overall_pbar = tqdm(
        total=total_tasks,
        desc="Overall Study",
        dynamic_ncols=True,
        bar_format=ETA_BAR_FORMAT,
        smoothing=0.1,
        position=0,
    )
    for experiment_cfg in experiment_cfgs:
        experiment_payloads.append(experiment_cfg.to_dict())
        rows: List[Dict[str, Any]] = []
        tasks = _build_tasks(experiment_cfg)
        num_workers = _resolve_num_workers(experiment_cfg)
        pbar = tqdm(
            total=len(tasks),
            desc=f"Experiment {experiment_cfg.experiment_id}",
            dynamic_ncols=True,
            bar_format=ETA_BAR_FORMAT,
            smoothing=0.1,
            position=1,
            leave=False,
        )
        if num_workers <= 1:
            for task in tasks:
                rows.extend(_evaluate_replicate_task(task))
                pbar.update(1)
                overall_pbar.update(1)
        else:
            ctx = get_context("spawn")
            with ctx.Pool(processes=num_workers) as pool:
                for result_rows in pool.imap_unordered(_evaluate_replicate_task, tasks, chunksize=max(1, int(experiment_cfg.mp_chunksize))):
                    rows.extend(result_rows)
                    pbar.update(1)
                    overall_pbar.update(1)
        pbar.close()
        experiment_frames.append(pd.DataFrame(rows))
    overall_pbar.close()
    raw_df = pd.concat(experiment_frames, ignore_index=True) if experiment_frames else pd.DataFrame()
    primary_level = float(config.overrides.get("intervals", {}).get("primary_level", 0.9)) if isinstance(config.overrides.get("intervals", {}), dict) else 0.9
    artifacts = _finalize_run(
        raw_df=raw_df,
        run_dir=run_dir,
        config_payload={"study": config.to_dict(), "experiments": experiment_payloads},
        output_subdir=config.output_subdir,
        primary_level=primary_level,
        elapsed_sec=perf_counter() - start,
    )
    return raw_df, run_dir, artifacts


def analyze_saved_results(results_path: str | Path, output_dir: str | Path | None = None, primary_level: float = 0.9) -> Dict[str, pd.DataFrame]:
    resolved = resolve_latest_path(results_path)
    df = pd.read_parquet(resolved) if resolved.suffix == ".parquet" else pd.read_csv(resolved)
    final_output = Path(output_dir) if output_dir is not None else resolved.parent / "artifacts"
    artifacts = generate_study_artifacts(df, output_dir=final_output, primary_level=primary_level)
    from ess_ope.study.plotting import generate_study_figures

    generate_study_figures(
        raw_df=df,
        point_df=artifacts["point_estimates"],
        table_2=artifacts["table_2_diagnostic_usefulness"],
        output_dir=final_output,
        primary_level=primary_level,
    )
    return artifacts
