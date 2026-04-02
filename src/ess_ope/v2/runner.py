from __future__ import annotations

from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from ess_ope.data.generate import generate_offline_dataset
from ess_ope.evaluation.ground_truth import dynamic_programming_value
from ess_ope.policies.tabular import TabularPolicy
from ess_ope.utils.logging import create_run_dir, refresh_latest_pointer, save_results_table, save_run_metadata, update_latest_pointer
from ess_ope.v2.analysis import generate_phase_artifacts
from ess_ope.v2.config import PhaseConfig, SuiteConfig
from ess_ope.v2.envs import build_environment
from ess_ope.v2.estimators import evaluate_estimator
from ess_ope.v2.inference import BootstrapSummary, episode_bootstrap, summarize_intervals
from ess_ope.v2.plotting import generate_v2_figures
from ess_ope.v2.policies import build_policy_pair


def _iter_conditions(config: PhaseConfig):
    for seed in config.seeds:
        for sample_size in config.sample_sizes:
            for mismatch_level in config.mismatch_levels:
                for reward_noise_level in config.reward_noise_levels:
                    for support_regime in config.support_regimes:
                        for horizon in config.horizons:
                            for reward_regime in config.reward_regimes:
                                for rarity_level in config.rarity_levels:
                                    yield {
                                        "seed": int(seed),
                                        "sample_size": int(sample_size),
                                        "mismatch_level": str(mismatch_level),
                                        "reward_noise_level": str(reward_noise_level),
                                        "support_regime": str(support_regime),
                                        "horizon": int(horizon),
                                        "reward_regime": str(reward_regime),
                                        "rarity_level": str(rarity_level),
                                    }


def _seed_hash(*values: Any) -> int:
    text = "|".join(map(str, values)).encode("utf-8")
    return int.from_bytes(text[:8].ljust(8, b"0"), "little") % (2**32 - 1)


def _point_row(
    config: PhaseConfig,
    condition: Dict[str, Any],
    replicate_id: int,
    truth_value: float,
    estimator_res,
    dataset_seed: int,
) -> Dict[str, Any]:
    return {
        "phase_name": config.name,
        "env_family": config.env_family,
        "condition_id": "|".join([config.name] + [f"{k}={condition[k]}" for k in sorted(condition)]),
        "replicate_id": int(replicate_id),
        "dataset_seed": int(dataset_seed),
        "estimator_key": estimator_res.estimator_key,
        "estimator_family": estimator_res.estimator_family,
        "ci_method": "point",
        "ci_level": np.nan,
        "estimate": float(estimator_res.estimate),
        "true_value": float(truth_value),
        "error": float(estimator_res.estimate - truth_value),
        "abs_error": float(abs(estimator_res.estimate - truth_value)),
        "squared_error": float((estimator_res.estimate - truth_value) ** 2),
        "ci_low": np.nan,
        "ci_high": np.nan,
        "ci_width": np.nan,
        "covered": np.nan,
        "ci_center_error": np.nan,
        "z_score_error": np.nan,
        "native_diagnostic_kind": estimator_res.native_diagnostic_kind,
        "native_diagnostic_value": estimator_res.native_diagnostic_value,
        "weight_cv": estimator_res.auxiliary.get("weight_cv", np.nan),
        "weight_entropy": estimator_res.auxiliary.get("weight_entropy", np.nan),
        "weight_perplexity": estimator_res.auxiliary.get("weight_perplexity", np.nan),
        "normalized_ess": estimator_res.auxiliary.get("normalized_ess", np.nan),
        "min_weight": estimator_res.auxiliary.get("min_weight", np.nan),
        "max_weight": estimator_res.auxiliary.get("max_weight", np.nan),
        "bootstrap_variance": np.nan,
        "estimator_runtime_sec": float(estimator_res.runtime_sec),
        "bootstrap_runtime_sec": 0.0,
        **condition,
    }


def _interval_rows(
    base_row: Dict[str, Any],
    intervals: Dict[str, Any],
    bootstrap: BootstrapSummary | None,
    truth_value: float,
) -> List[Dict[str, Any]]:
    rows = []
    for method, interval in intervals.items():
        row = dict(base_row)
        row["ci_method"] = method
        row["ci_low"] = float(interval.low)
        row["ci_high"] = float(interval.high)
        row["ci_width"] = float(interval.width)
        row["covered"] = float(interval.low <= truth_value <= interval.high) if np.isfinite(interval.low) and np.isfinite(interval.high) else np.nan
        row["ci_center_error"] = float(interval.center - truth_value) if np.isfinite(interval.center) else np.nan
        if method == "analytic" and np.isfinite(interval.width) and interval.width > 0:
            approx_se = interval.width / (2.0 * 1.96)
            row["z_score_error"] = float(base_row["error"] / max(1e-12, approx_se))
        row["bootstrap_variance"] = float(bootstrap.variance) if bootstrap is not None else np.nan
        row["bootstrap_runtime_sec"] = float(bootstrap.runtime_sec) if bootstrap is not None else 0.0
        rows.append(row)
    return rows


def _evaluate_condition_rows(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    cfg = PhaseConfig.from_dict(task["config"])
    condition = task["condition"]
    rows: List[Dict[str, Any]] = []

    bundle = build_environment(
        env_family=cfg.env_family,
        seed=int(condition["seed"]),
        sample_size=int(condition["sample_size"]),
        horizon=int(condition["horizon"]),
        reward_noise_level=str(condition["reward_noise_level"]),
        rarity_level=str(condition["rarity_level"]),
        env_cfg=cfg.env,
    )
    env = bundle.env
    # Use a soft reference policy derived from approximate rewards to define the evaluation policy.
    ref_logits = env.rewards.copy()
    if ref_logits.shape[0] == 1:
        ref_probs = np.exp(ref_logits[0] - np.max(ref_logits[0], axis=-1, keepdims=True))
        ref_probs /= np.sum(ref_probs, axis=-1, keepdims=True)
    else:
        ref_probs = np.exp(ref_logits - np.max(ref_logits, axis=-1, keepdims=True))
        ref_probs /= np.sum(ref_probs, axis=-1, keepdims=True)
    from ess_ope.policies.tabular import TabularPolicy

    reference_policy = TabularPolicy(ref_probs if env.horizon > 1 else ref_probs)
    truth_seed_policy = dynamic_programming_value(env, reference_policy, gamma=cfg.gamma)
    policies = build_policy_pair(
        truth=truth_seed_policy,
        mismatch_level=str(condition["mismatch_level"]),
        support_regime=str(condition["support_regime"]),
        seed=int(condition["seed"]) + 17,
    )
    truth = dynamic_programming_value(env, policies.target_policy, gamma=cfg.gamma)

    estimators = list(cfg.estimators)
    if cfg.include_reference_estimators:
        for key in ["dr_oracle", "fqe_tabular"]:
            if key not in estimators:
                estimators.append(key)

    for replicate_id in range(int(cfg.replicates)):
        data_seed = _seed_hash(cfg.name, condition["seed"], replicate_id, condition["sample_size"], condition["mismatch_level"], condition["support_regime"], condition["rarity_level"])
        dataset = generate_offline_dataset(
            env=env,
            behavior_policy=policies.behavior_policy,
            num_episodes=int(condition["sample_size"]),
            horizon=env.horizon,
            seed=data_seed,
        )
        for estimator_key in estimators:
            estimator_res = evaluate_estimator(
                estimator_key=estimator_key,
                dataset=dataset,
                target_policy=policies.target_policy,
                behavior_policy=policies.behavior_policy,
                env=env,
                gamma=cfg.gamma,
                truth_q=truth.q,
                truth_v=truth.v,
                l2_reg=float(cfg.env.get("fqe_l2_reg", 1e-4)),
            )
            point_row = _point_row(cfg, condition, replicate_id, truth.value, estimator_res, data_seed)
            point_row["policy_mismatch_mix"] = float(policies.metadata["mismatch_mix"])
            point_row["policy_support_floor"] = float(policies.metadata["support_floor"])
            rows.append(point_row)

            bootstrap = None
            if any(method.startswith("bootstrap") for method in cfg.ci_methods) and cfg.bootstrap_samples > 0:
                bootstrap = episode_bootstrap(
                    dataset=dataset,
                    estimate_fn=lambda ds, key=estimator_key: evaluate_estimator(
                        estimator_key=key,
                        dataset=ds,
                        target_policy=policies.target_policy,
                        behavior_policy=policies.behavior_policy,
                        env=env,
                        gamma=cfg.gamma,
                        truth_q=truth.q,
                        truth_v=truth.v,
                        l2_reg=float(cfg.env.get("fqe_l2_reg", 1e-4)),
                    ).estimate,
                    n_boot=int(cfg.bootstrap_samples),
                    seed=_seed_hash(data_seed, estimator_key, "bootstrap"),
                    subsample_ratio=float(cfg.bootstrap_subsample_ratio),
                )

            for ci_level in cfg.ci_levels:
                intervals = summarize_intervals(
                    point_estimate=estimator_res.estimate,
                    contributions=estimator_res.episode_contributions if "analytic" in cfg.ci_methods else None,
                    bootstrap=bootstrap if any(method.startswith("bootstrap") for method in cfg.ci_methods) else None,
                    ci_level=float(ci_level),
                )
                for interval_row in _interval_rows(point_row | {"ci_level": float(ci_level)}, intervals, bootstrap, truth.value):
                    if interval_row["ci_method"] in cfg.ci_methods:
                        rows.append(interval_row)

    return rows


def analyze_phase_results(raw_df: pd.DataFrame, output_dir: str | Path) -> Dict[str, pd.DataFrame]:
    artifacts = generate_phase_artifacts(raw_df, output_dir=output_dir)
    generate_v2_figures(
        point_df=artifacts["point_estimates"],
        calibration_summary=artifacts["calibration_summary"],
        table_a=artifacts["table_a_estimator_summary"],
        table_b=artifacts["table_b_diagnostic_quality"],
        cross_ranking=artifacts["cross_estimator_ranking"],
        diagnostic_corr=artifacts["diagnostic_correlation_summary"],
        output_dir=output_dir,
    )
    return artifacts


def run_phase(config: PhaseConfig) -> Tuple[pd.DataFrame, Path, Dict[str, pd.DataFrame]]:
    rows: List[Dict[str, Any]] = []
    total = (
        len(config.seeds)
        * len(config.sample_sizes)
        * len(config.mismatch_levels)
        * len(config.reward_noise_levels)
        * len(config.support_regimes)
        * len(config.horizons)
        * len(config.reward_regimes)
        * len(config.rarity_levels)
    )
    num_workers = int(config.num_workers)
    pbar = tqdm(total=total, desc=f"V2:{config.name}", dynamic_ncols=True)

    tasks = [{"config": config.to_dict(), "condition": condition} for condition in _iter_conditions(config)]
    if num_workers <= 1:
        for task in tasks:
            rows.extend(_evaluate_condition_rows(task))
            pbar.update(1)
    else:
        ctx = get_context("spawn")
        with ctx.Pool(processes=num_workers) as pool:
            for result_rows in pool.imap_unordered(_evaluate_condition_rows, tasks, chunksize=max(1, int(config.mp_chunksize))):
                rows.extend(result_rows)
                pbar.update(1)
    pbar.close()

    raw_df = pd.DataFrame(rows)
    run_dir = create_run_dir(config.results_root, config.name)
    save_run_metadata(run_dir, config.to_dict())
    save_results_table(raw_df, run_dir, "replicate_results")
    artifacts_dir = run_dir / config.output_subdir
    artifacts = analyze_phase_results(raw_df, artifacts_dir)
    update_latest_pointer(config.results_root, run_dir)
    refresh_latest_pointer(config.results_root)
    return raw_df, run_dir, artifacts


def run_suite(config: SuiteConfig) -> Tuple[pd.DataFrame, Path]:
    run_dir = create_run_dir(config.results_root, config.name)
    save_run_metadata(run_dir, config.to_dict())
    phase_frames: List[pd.DataFrame] = []

    for phase_path in config.phase_configs:
        phase_cfg = PhaseConfig.from_yaml(phase_path)
        phase_cfg.results_root = str(run_dir)
        phase_df, _, _ = run_phase(phase_cfg)
        phase_frames.append(phase_df)

    combined = pd.concat(phase_frames, ignore_index=True) if phase_frames else pd.DataFrame()
    save_results_table(combined, run_dir, "replicate_results")
    analyze_phase_results(combined, run_dir / config.output_subdir)
    update_latest_pointer(config.results_root, run_dir)
    refresh_latest_pointer(config.results_root)
    return combined, run_dir
