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
from ess_ope.estimators.dr import doubly_robust_estimate
from ess_ope.estimators import is_family_estimates
from ess_ope.estimators.mrdr import mrdr_linear_estimate
from ess_ope.envs.random_mdp import RandomMDP, RandomMDPConfig
from ess_ope.evaluation.ground_truth import dynamic_programming_value
from ess_ope.metrics.errors import point_error_metrics
from ess_ope.metrics.ess import weight_summary
from ess_ope.policies.softmax import softmax_policy_from_logits, statewise_kl
from ess_ope.utils.logging import create_run_dir, save_results_table, save_run_metadata, update_latest_pointer


@dataclass
class SweepConfig:
    name: str = "random_mdp_baseline"
    results_root: str = "results"
    gamma: float = 1.0
    temperature: float = 1.0
    seeds: List[int] = field(default_factory=lambda: list(range(10)))
    alphas: List[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0])
    betas: List[float] = field(default_factory=lambda: [0.0, 0.5, 1.0])
    dataset_sizes: List[int] = field(default_factory=lambda: [50, 200, 1000])
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

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SweepConfig":
        base = cls()
        for key, value in payload.items():
            if hasattr(base, key):
                setattr(base, key, value)
        return base

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _make_env(config: SweepConfig, beta: float, seed: int) -> RandomMDP:
    env_kwargs = dict(config.env)
    env_kwargs["beta"] = beta
    env_kwargs["seed"] = int(seed)
    env_cfg = RandomMDPConfig(**env_kwargs)
    return RandomMDP.generate(env_cfg)


def _policy_pair(num_states: int, num_actions: int, alpha: float, temperature: float, seed: int):
    rng = np.random.default_rng(seed)
    theta_target = rng.normal(size=(num_states, num_actions))
    theta_random = rng.normal(size=(num_states, num_actions))
    theta_behavior = (1.0 - alpha) * theta_target + alpha * theta_random

    target = softmax_policy_from_logits(theta_target, temperature=temperature)
    behavior = softmax_policy_from_logits(theta_behavior, temperature=temperature)
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

    env_seed = _seed_hash(seed, beta, env_repeat_id, "env")
    env_kwargs = dict(cfg["env"])
    env_kwargs["beta"] = beta
    env_kwargs["seed"] = int(env_seed)
    env = RandomMDP.generate(RandomMDPConfig(**env_kwargs))

    policy_seed = _seed_hash(seed, beta, env_repeat_id, policy_repeat_id, "policy")
    target_policy, behavior_policy = _policy_pair(
        num_states=env.num_states,
        num_actions=env.num_actions,
        alpha=alpha,
        temperature=float(cfg["temperature"]),
        seed=policy_seed,
    )
    kl_vals = statewise_kl(target_policy, behavior_policy)
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

    return [row]


def _iter_tasks(config: SweepConfig) -> Any:
    cfg_payload = config.to_dict()
    for seed in config.seeds:
        for beta in config.betas:
            for env_repeat_id in range(int(config.env_repeats)):
                for policy_repeat_id in range(int(config.policy_repeats)):
                    for alpha in config.alphas:
                        for k in config.dataset_sizes:
                            for repeat_id in range(int(config.dataset_repeats)):
                                yield {
                                    "cfg": cfg_payload,
                                    "seed": int(seed),
                                    "beta": float(beta),
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
