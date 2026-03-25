from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from ess_ope.data.generate import generate_offline_dataset
from ess_ope.estimators.dm_fqe import direct_model_tabular, fitted_q_evaluation
from ess_ope.estimators.dr import doubly_robust_estimate, doubly_robust_episode_contributions
from ess_ope.estimators import is_family_estimates
from ess_ope.estimators.mrdr import mrdr_linear_estimate
from ess_ope.envs.chain_bandit import ChainBanditConfig, ChainBanditEnv
from ess_ope.envs.random_mdp import RandomMDP, RandomMDPConfig
from ess_ope.evaluation.ground_truth import dynamic_programming_value
from ess_ope.metrics.confidence import IntervalEstimate, bootstrap_estimator_interval, wald_mean_interval
from ess_ope.metrics.errors import point_error_metrics
from ess_ope.metrics.ess import weight_summary
from ess_ope.policies.tabular import TabularPolicy
from ess_ope.utils.logging import create_run_dir, save_results_table, save_run_metadata, update_latest_pointer


@dataclass
class SweepConfig:
    name: str = "random_mdp_baseline"
    env_name: str = "random_mdp"
    results_root: str = "results"
    gamma: float = 1.0
    temperature: float = 1.0
    seeds: List[int] = field(default_factory=lambda: list(range(10)))
    alphas: List[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0])
    betas: List[float] = field(default_factory=lambda: [0.0, 0.5, 1.0])
    dataset_sizes: List[int] = field(default_factory=lambda: [50, 200, 1000])
    transition_strengths: List[float] = field(default_factory=lambda: [0.5])
    reward_mean_scales: List[float] = field(default_factory=lambda: [1.0])
    reward_gaps: List[float] = field(default_factory=lambda: [0.5])
    reward_stds: List[float] = field(default_factory=lambda: [0.5])
    chain_variants: List[str] = field(default_factory=lambda: ["transitional"])
    env_repeats: int = 1
    policy_repeats: int = 1
    dataset_repeats: int = 1
    num_workers: int = 1
    mp_chunksize: int = 1
    env: Dict[str, Any] = field(
        default_factory=lambda: {
            "num_states": 100,
            "num_actions": 6,
            "horizon": 20,
            "branch_factor": 3,
            "linear_feature_dim": 12,
            "reward_noise_std": 0.0,
        }
    )
    fqe_l2_reg: float = 1e-4
    mrdr_l2_reg: float = 1e-3
    analysis_estimators: List[str] = field(
        default_factory=lambda: ["is_pdis", "dr_oracle", "dm_tabular", "fqe_linear"]
    )
    interval_mode: str = "none"
    ci_level: float = 0.95
    ci_bootstrap_samples: int = 0

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SweepConfig":
        base = cls()
        for key, value in payload.items():
            if hasattr(base, key):
                setattr(base, key, value)
        return base

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _make_env(config: SweepConfig, beta: float, seed: int):
    task = {
        "cfg": config.to_dict(),
        "beta": float(beta),
        "env_seed": int(seed),
        "transition_strength": float(config.transition_strengths[0]),
        "reward_mean_scale": float(config.reward_mean_scales[0]),
        "reward_gap": float(config.reward_gaps[0]),
        "reward_std": float(config.reward_stds[0]),
        "chain_variant": str(config.chain_variants[0]),
    }
    return _make_env_from_task(task)


def _softmax_policy_from_logits(logits: np.ndarray, temperature: float) -> TabularPolicy:
    scaled = np.asarray(logits, dtype=float) / max(temperature, 1e-8)
    shifted = scaled - np.max(scaled, axis=-1, keepdims=True)
    probs = np.exp(shifted)
    probs = probs / np.sum(probs, axis=-1, keepdims=True)
    return TabularPolicy(probs)


def _statewise_kl(target_policy: TabularPolicy, behavior_policy: TabularPolicy) -> np.ndarray:
    p = np.clip(target_policy.probs, 1e-12, 1.0)
    q = np.clip(behavior_policy.probs, 1e-12, 1.0)
    kls = np.sum(p * (np.log(p) - np.log(q)), axis=-1)
    return np.asarray(kls, dtype=float).reshape(-1)


def _make_env_from_task(task: Dict[str, Any]):
    cfg = task["cfg"]
    beta = float(task["beta"])
    env_seed = int(task["env_seed"])
    env_kwargs = dict(cfg["env"])

    if cfg.get("env_name", "random_mdp") == "chain_bandit":
        env_cfg = ChainBanditConfig(
            num_states=int(env_kwargs["num_states"]),
            num_actions=int(env_kwargs["num_actions"]),
            horizon=int(env_kwargs["horizon"]),
            linear_feature_dim=int(env_kwargs.get("linear_feature_dim", 12)),
            transition_strength=float(task["transition_strength"]),
            reward_mean_scale=float(task["reward_mean_scale"]),
            reward_gap=float(task["reward_gap"]),
            reward_std=float(task["reward_std"]),
            beta=beta,
            variant=str(task["chain_variant"]),
            seed=env_seed,
        )
        return ChainBanditEnv.generate(env_cfg)

    env_kwargs["beta"] = beta
    env_kwargs["seed"] = env_seed
    env_cfg = RandomMDPConfig(**env_kwargs)
    return RandomMDP.generate(env_cfg)


def _policy_pair(
    num_states: int,
    num_actions: int,
    alpha: float,
    temperature: float,
    seed: int,
    horizon: int | None = None,
):
    rng = np.random.default_rng(seed)
    logits_shape = (num_states, num_actions) if horizon is None else (horizon, num_states, num_actions)
    theta_target = rng.normal(size=logits_shape)
    theta_random = rng.normal(size=logits_shape)
    theta_behavior = (1.0 - alpha) * theta_target + alpha * theta_random

    target = _softmax_policy_from_logits(theta_target, temperature=temperature)
    behavior = _softmax_policy_from_logits(theta_behavior, temperature=temperature)
    return target, behavior


def _seed_hash(*values: Any) -> int:
    text = "|".join(map(str, values)).encode("utf-8")
    digest = hashlib.blake2b(text, digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32 - 1)


def _evaluate_condition(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    cfg = task["cfg"]
    seed = int(task["seed"])
    beta = float(task["beta"])
    env_repeat_id = int(task["env_repeat_id"])
    policy_repeat_id = int(task["policy_repeat_id"])
    alpha = float(task["alpha"])
    k = int(task["k"])
    repeat_id = int(task["repeat_id"])
    transition_strength = float(task["transition_strength"])
    reward_mean_scale = float(task["reward_mean_scale"])
    reward_gap = float(task["reward_gap"])
    reward_std = float(task["reward_std"])
    chain_variant = str(task["chain_variant"])

    env_seed = _seed_hash(seed, beta, env_repeat_id, "env")
    task = dict(task)
    task["env_seed"] = env_seed
    env = _make_env_from_task(task)

    policy_seed = _seed_hash(seed, beta, env_repeat_id, policy_repeat_id, "policy")
    target_policy, behavior_policy = _policy_pair(
        num_states=env.num_states,
        num_actions=env.num_actions,
        alpha=alpha,
        temperature=float(cfg["temperature"]),
        seed=policy_seed,
        horizon=env.horizon if cfg.get("env_name", "random_mdp") == "chain_bandit" else None,
    )
    kl_vals = _statewise_kl(target_policy, behavior_policy)
    truth = dynamic_programming_value(env, target_policy, gamma=float(cfg["gamma"]))

    data_seed = _seed_hash(
        seed,
        beta,
        env_repeat_id,
        policy_repeat_id,
        alpha,
        k,
        repeat_id,
        "dataset",
    )
    dataset = generate_offline_dataset(
        env=env,
        behavior_policy=behavior_policy,
        num_episodes=int(k),
        horizon=env.horizon,
        seed=data_seed,
    )

    is_res = is_family_estimates(
        dataset=dataset,
        target_policy=target_policy,
        behavior_policy=behavior_policy,
        gamma=float(cfg["gamma"]),
    )
    w_stats = weight_summary(np.asarray(is_res["episode_weights"]))
    is_contrib = np.asarray(
        np.sum(
            np.asarray(is_res["partial_weights"], dtype=float)
            * dataset.rewards
            * (float(cfg["gamma"]) ** np.arange(env.horizon))[None, :],
            axis=1,
        ),
        dtype=float,
    )

    dm_res = direct_model_tabular(
        dataset=dataset,
        target_policy=target_policy,
        num_states=env.num_states,
        num_actions=env.num_actions,
        horizon=env.horizon,
        initial_state_dist=env.initial_state_dist,
        gamma=float(cfg["gamma"]),
    )

    fqe_tab = fitted_q_evaluation(
        dataset=dataset,
        target_policy=target_policy,
        num_states=env.num_states,
        num_actions=env.num_actions,
        horizon=env.horizon,
        initial_state_dist=env.initial_state_dist,
        model_type="tabular",
        gamma=float(cfg["gamma"]),
        l2_reg=float(cfg["fqe_l2_reg"]),
    )

    fqe_lin = fitted_q_evaluation(
        dataset=dataset,
        target_policy=target_policy,
        num_states=env.num_states,
        num_actions=env.num_actions,
        horizon=env.horizon,
        initial_state_dist=env.initial_state_dist,
        model_type="linear",
        feature_tensor=env.linear_sa_features,
        gamma=float(cfg["gamma"]),
        l2_reg=float(cfg["fqe_l2_reg"]),
    )

    dr_value = doubly_robust_estimate(
        dataset=dataset,
        target_policy=target_policy,
        behavior_policy=behavior_policy,
        q_hat=fqe_lin.q,
        v_hat=fqe_lin.v,
        gamma=float(cfg["gamma"]),
    )
    dr_oracle_contrib = doubly_robust_episode_contributions(
        dataset=dataset,
        target_policy=target_policy,
        behavior_policy=behavior_policy,
        q_hat=truth.q,
        v_hat=truth.v,
        gamma=float(cfg["gamma"]),
    )
    dr_oracle_value = float(np.mean(dr_oracle_contrib))

    mrdr = mrdr_linear_estimate(
        dataset=dataset,
        target_policy=target_policy,
        behavior_policy=behavior_policy,
        feature_tensor=env.linear_sa_features,
        num_states=env.num_states,
        num_actions=env.num_actions,
        horizon=env.horizon,
        initial_state_dist=env.initial_state_dist,
        gamma=float(cfg["gamma"]),
        l2_reg=float(cfg["mrdr_l2_reg"]),
    )

    estimates = {
        "is_trajectory": float(is_res["is_trajectory"]),
        "wis_trajectory": float(is_res["wis_trajectory"]),
        "is_pdis": float(is_res["is_pdis"]),
        "wis_pdis": float(is_res["wis_pdis"]),
        "dr_oracle": dr_oracle_value,
        "dm_tabular": dm_res.value,
        "fqe_tabular": fqe_tab.value,
        "fqe_linear": fqe_lin.value,
        "dr": dr_value,
        "mrdr": mrdr.value,
    }

    row: Dict[str, Any] = {
        "env_id": env.env_id,
        "seed": int(seed),
        "env_repeat_id": int(env_repeat_id),
        "policy_repeat_id": int(policy_repeat_id),
        "repeat_id": int(repeat_id),
        "env_seed": int(env_seed),
        "policy_seed": int(policy_seed),
        "dataset_seed": int(data_seed),
        "alpha": float(alpha),
        "beta": float(beta),
        "K": int(k),
        "env_name": str(cfg.get("env_name", "random_mdp")),
        "transition_strength": transition_strength,
        "reward_mean_scale": reward_mean_scale,
        "reward_gap": reward_gap,
        "reward_std": reward_std,
        "chain_variant": chain_variant,
        "H": int(env.horizon),
        "num_states": int(env.num_states),
        "num_actions": int(env.num_actions),
        "state_kl_mean": float(np.mean(kl_vals)),
        "state_kl_std": float(np.std(kl_vals)),
        "v_true": truth.value,
        **w_stats,
        **{f"estimate_{k_est}": v for k_est, v in estimates.items()},
    }
    row["ess_is_over_k"] = float(row["ess_is"] / max(1, row["K"]))

    for name, estimate in estimates.items():
        errs = point_error_metrics(estimate, truth.value)
        row[f"error_{name}"] = errs["error"]
        row[f"abs_error_{name}"] = errs["abs_error"]
        row[f"squared_error_{name}"] = errs["squared_error"]

    interval_mode = str(cfg.get("interval_mode", "none")).lower()
    selected_estimators = {
        str(key)
        for key in cfg.get("analysis_estimators", ["is_pdis", "dr_oracle", "dm_tabular", "fqe_linear"])
    }

    def _store_interval(method: str, key: str, interval: IntervalEstimate) -> None:
        row[f"ci_{method}_low_{key}"] = interval.low
        row[f"ci_{method}_high_{key}"] = interval.high
        row[f"ci_{method}_width_{key}"] = interval.width
        if np.isfinite(interval.low) and np.isfinite(interval.high):
            covered = float(interval.low <= truth.value <= interval.high)
        else:
            covered = np.nan
        row[f"ci_{method}_covered_{key}"] = covered

    analytic_contribs = {
        "is_pdis": is_contrib,
        "dr_oracle": dr_oracle_contrib,
    }
    if interval_mode in {"analytic", "both"}:
        for key in selected_estimators:
            contrib = analytic_contribs.get(key)
            if contrib is None:
                _store_interval("analytic", key, IntervalEstimate(np.nan, np.nan, np.nan, np.nan))
                continue
            _store_interval("analytic", key, wald_mean_interval(contrib, ci_level=float(cfg["ci_level"])))

    if interval_mode in {"bootstrap", "both"} and int(cfg.get("ci_bootstrap_samples", 0)) > 0:
        gamma = float(cfg["gamma"])
        bootstrap_fns = {
            "is_pdis": lambda ds: float(
                is_family_estimates(
                    dataset=ds,
                    target_policy=target_policy,
                    behavior_policy=behavior_policy,
                    gamma=gamma,
                )["is_pdis"]
            ),
            "dr_oracle": lambda ds: doubly_robust_estimate(
                dataset=ds,
                target_policy=target_policy,
                behavior_policy=behavior_policy,
                q_hat=truth.q,
                v_hat=truth.v,
                gamma=gamma,
            ),
            "dm_tabular": lambda ds: direct_model_tabular(
                dataset=ds,
                target_policy=target_policy,
                num_states=env.num_states,
                num_actions=env.num_actions,
                horizon=env.horizon,
                initial_state_dist=env.initial_state_dist,
                gamma=gamma,
            ).value,
            "fqe_linear": lambda ds: fitted_q_evaluation(
                dataset=ds,
                target_policy=target_policy,
                num_states=env.num_states,
                num_actions=env.num_actions,
                horizon=env.horizon,
                initial_state_dist=env.initial_state_dist,
                model_type="linear",
                feature_tensor=env.linear_sa_features,
                gamma=gamma,
                l2_reg=float(cfg["fqe_l2_reg"]),
            ).value,
        }

        for idx, key in enumerate(sorted(selected_estimators)):
            fn = bootstrap_fns.get(key)
            if fn is None:
                _store_interval("bootstrap", key, IntervalEstimate(np.nan, np.nan, np.nan, np.nan))
                continue
            boot = bootstrap_estimator_interval(
                dataset=dataset,
                estimate_fn=fn,
                n_boot=int(cfg["ci_bootstrap_samples"]),
                ci_level=float(cfg["ci_level"]),
                seed=_seed_hash(data_seed, key, idx, "bootstrap"),
            )
            _store_interval("bootstrap", key, boot)

    return [row]


def _iter_tasks(config: SweepConfig) -> Any:
    cfg_payload = config.to_dict()
    for seed in config.seeds:
        for beta in config.betas:
            for transition_strength in config.transition_strengths:
                for reward_mean_scale in config.reward_mean_scales:
                    for reward_gap in config.reward_gaps:
                        for reward_std in config.reward_stds:
                            for chain_variant in config.chain_variants:
                                for env_repeat_id in range(int(config.env_repeats)):
                                    for policy_repeat_id in range(int(config.policy_repeats)):
                                        for alpha in config.alphas:
                                            for k in config.dataset_sizes:
                                                for repeat_id in range(int(config.dataset_repeats)):
                                                    yield {
                                                        "cfg": cfg_payload,
                                                        "seed": int(seed),
                                                        "beta": float(beta),
                                                        "transition_strength": float(transition_strength),
                                                        "reward_mean_scale": float(reward_mean_scale),
                                                        "reward_gap": float(reward_gap),
                                                        "reward_std": float(reward_std),
                                                        "chain_variant": str(chain_variant),
                                                        "env_repeat_id": int(env_repeat_id),
                                                        "policy_repeat_id": int(policy_repeat_id),
                                                        "alpha": float(alpha),
                                                        "k": int(k),
                                                        "repeat_id": int(repeat_id),
                                                    }


def run_sweep(config: SweepConfig) -> Tuple[pd.DataFrame, Path, Path]:
    rows: List[Dict[str, Any]] = []

    total = (
        len(config.seeds)
        * int(config.env_repeats)
        * int(config.policy_repeats)
        * len(config.betas)
        * len(config.transition_strengths)
        * len(config.reward_mean_scales)
        * len(config.reward_gaps)
        * len(config.reward_stds)
        * len(config.chain_variants)
        * len(config.alphas)
        * len(config.dataset_sizes)
        * int(config.dataset_repeats)
    )
    pbar = tqdm(total=total, desc=f"Sweep:{config.name}")

    if int(config.num_workers) <= 1:
        for task in _iter_tasks(config):
            condition_rows = _evaluate_condition(task)
            rows.extend(condition_rows)
            pbar.update(len(condition_rows))
    else:
        ctx = get_context("spawn")
        with ctx.Pool(processes=int(config.num_workers)) as pool:
            for condition_rows in pool.imap_unordered(
                _evaluate_condition,
                _iter_tasks(config),
                chunksize=max(1, int(config.mp_chunksize)),
            ):
                rows.extend(condition_rows)
                pbar.update(len(condition_rows))

    pbar.close()

    df = pd.DataFrame(rows)
    run_dir = create_run_dir(config.results_root, config.name)
    save_run_metadata(run_dir, config.to_dict())
    result_path = save_results_table(df, run_dir, stem="sweep_results")
    update_latest_pointer(config.results_root, run_dir)
    return df, run_dir, result_path
