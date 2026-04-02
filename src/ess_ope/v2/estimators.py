from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, Dict

import numpy as np

from ess_ope.data.dataset import EpisodeDataset
from ess_ope.estimators import compute_importance_weights, is_family_estimates
from ess_ope.estimators.dm_fqe import FQEResult, fitted_q_evaluation
from ess_ope.estimators.dr import doubly_robust_episode_contributions
from ess_ope.metrics.ess import normalized_weights, weight_summary
from ess_ope.policies.tabular import TabularPolicy


@dataclass
class EstimatorResult:
    estimator_key: str
    estimator_family: str
    estimate: float
    native_diagnostic_kind: str | None
    native_diagnostic_value: float | None
    episode_contributions: np.ndarray | None = None
    auxiliary: Dict[str, Any] = field(default_factory=dict)
    runtime_sec: float = 0.0


def weighted_dr_episode_contributions(
    dataset: EpisodeDataset,
    target_policy: TabularPolicy,
    behavior_policy: TabularPolicy,
    q_hat: np.ndarray,
    v_hat: np.ndarray,
    gamma: float = 1.0,
    min_prob: float = 1e-12,
) -> np.ndarray:
    weights = compute_importance_weights(
        dataset=dataset,
        target_policy=target_policy,
        behavior_policy=behavior_policy,
        min_prob=min_prob,
    )
    k, h = dataset.num_episodes, dataset.horizon
    contrib = np.zeros(k, dtype=float)
    initial_vals = v_hat[0, dataset.states[:, 0]]
    contrib += np.asarray(initial_vals, dtype=float)

    for t in range(h):
        step_weights = normalized_weights(weights.partial_weights[:, t])
        for ep in range(k):
            s = dataset.states[ep, t]
            a = dataset.actions[ep, t]
            r = dataset.rewards[ep, t]
            sp = dataset.next_states[ep, t]
            done = dataset.dones[ep, t]
            td = r + gamma * (1.0 - float(done)) * v_hat[t + 1, sp] - q_hat[t, s, a]
            contrib[ep] += k * step_weights[ep] * td
    return contrib


def _trajectory_wis_contributions(dataset: EpisodeDataset, episode_weights: np.ndarray, gamma: float) -> np.ndarray:
    discounts = gamma ** np.arange(dataset.horizon)
    returns = np.sum(dataset.rewards * discounts[None, :], axis=1)
    norm = normalized_weights(episode_weights)
    return float(dataset.num_episodes) * norm * returns


def _fit_linear_fqe(
    dataset: EpisodeDataset,
    target_policy: TabularPolicy,
    env: Any,
    gamma: float,
    l2_reg: float,
) -> FQEResult:
    return fitted_q_evaluation(
        dataset=dataset,
        target_policy=target_policy,
        num_states=env.num_states,
        num_actions=env.num_actions,
        horizon=env.horizon,
        initial_state_dist=env.initial_state_dist,
        model_type="linear",
        feature_tensor=env.linear_sa_features,
        gamma=gamma,
        l2_reg=l2_reg,
    )


def _fit_tabular_fqe(
    dataset: EpisodeDataset,
    target_policy: TabularPolicy,
    env: Any,
    gamma: float,
    l2_reg: float,
) -> FQEResult:
    return fitted_q_evaluation(
        dataset=dataset,
        target_policy=target_policy,
        num_states=env.num_states,
        num_actions=env.num_actions,
        horizon=env.horizon,
        initial_state_dist=env.initial_state_dist,
        model_type="tabular",
        gamma=gamma,
        l2_reg=l2_reg,
    )


def evaluate_estimator(
    estimator_key: str,
    dataset: EpisodeDataset,
    target_policy: TabularPolicy,
    behavior_policy: TabularPolicy,
    env: Any,
    gamma: float,
    truth_q: np.ndarray | None = None,
    truth_v: np.ndarray | None = None,
    l2_reg: float = 1e-4,
) -> EstimatorResult:
    start = perf_counter()
    is_res = None
    if estimator_key in {"is", "wis", "pdis"}:
        is_res = is_family_estimates(dataset, target_policy, behavior_policy, gamma=gamma)

    if estimator_key == "is":
        episode_weights = np.asarray(is_res["episode_weights"], dtype=float)
        discounts = gamma ** np.arange(dataset.horizon)
        returns = np.sum(dataset.rewards * discounts[None, :], axis=1)
        contributions = episode_weights * returns
        wstats = weight_summary(episode_weights)
        return EstimatorResult(
            estimator_key="is",
            estimator_family="is_like",
            estimate=float(is_res["is_trajectory"]),
            native_diagnostic_kind="ess",
            native_diagnostic_value=float(wstats["ess_is"]),
            episode_contributions=contributions,
            auxiliary=wstats | {"normalized_ess": float(wstats["ess_is"] / max(1, dataset.num_episodes)), "min_weight": float(np.min(episode_weights)), "max_weight": float(np.max(episode_weights))},
            runtime_sec=perf_counter() - start,
        )

    if estimator_key == "wis":
        episode_weights = np.asarray(is_res["episode_weights"], dtype=float)
        contributions = _trajectory_wis_contributions(dataset, episode_weights, gamma=gamma)
        wstats = weight_summary(episode_weights)
        return EstimatorResult(
            estimator_key="wis",
            estimator_family="is_like",
            estimate=float(is_res["wis_trajectory"]),
            native_diagnostic_kind="ess",
            native_diagnostic_value=float(wstats["ess_is"]),
            episode_contributions=contributions,
            auxiliary=wstats | {"normalized_ess": float(wstats["ess_is"] / max(1, dataset.num_episodes)), "min_weight": float(np.min(episode_weights)), "max_weight": float(np.max(episode_weights))},
            runtime_sec=perf_counter() - start,
        )

    if estimator_key == "pdis":
        partial_weights = np.asarray(is_res["partial_weights"], dtype=float)
        discounts = gamma ** np.arange(dataset.horizon)
        contributions = np.sum(partial_weights * dataset.rewards * discounts[None, :], axis=1)
        episode_weights = np.asarray(is_res["episode_weights"], dtype=float)
        wstats = weight_summary(episode_weights)
        return EstimatorResult(
            estimator_key="pdis",
            estimator_family="is_like",
            estimate=float(is_res["is_pdis"]),
            native_diagnostic_kind="ess",
            native_diagnostic_value=float(wstats["ess_is"]),
            episode_contributions=contributions,
            auxiliary=wstats | {"normalized_ess": float(wstats["ess_is"] / max(1, dataset.num_episodes)), "min_weight": float(np.min(episode_weights)), "max_weight": float(np.max(episode_weights))},
            runtime_sec=perf_counter() - start,
        )

    if estimator_key in {"dr", "wdr", "fqe_linear", "fqe_tabular"}:
        if estimator_key == "fqe_tabular":
            fqe = _fit_tabular_fqe(dataset, target_policy, env, gamma=gamma, l2_reg=l2_reg)
        else:
            fqe = _fit_linear_fqe(dataset, target_policy, env, gamma=gamma, l2_reg=l2_reg)

        if estimator_key == "fqe_linear":
            return EstimatorResult(
                estimator_key="fqe_linear",
                estimator_family="value_model",
                estimate=float(fqe.value),
                native_diagnostic_kind=None,
                native_diagnostic_value=None,
                episode_contributions=None,
                auxiliary={},
                runtime_sec=perf_counter() - start,
            )
        if estimator_key == "fqe_tabular":
            return EstimatorResult(
                estimator_key="fqe_tabular",
                estimator_family="reference_model",
                estimate=float(fqe.value),
                native_diagnostic_kind=None,
                native_diagnostic_value=None,
                episode_contributions=None,
                auxiliary={},
                runtime_sec=perf_counter() - start,
            )

        if estimator_key == "dr":
            contributions = doubly_robust_episode_contributions(
                dataset=dataset,
                target_policy=target_policy,
                behavior_policy=behavior_policy,
                q_hat=fqe.q,
                v_hat=fqe.v,
                gamma=gamma,
            )
            return EstimatorResult(
                estimator_key="dr",
                estimator_family="doubly_robust",
                estimate=float(np.mean(contributions)),
                native_diagnostic_kind=None,
                native_diagnostic_value=None,
                episode_contributions=contributions,
                auxiliary={},
                runtime_sec=perf_counter() - start,
            )

        contributions = weighted_dr_episode_contributions(
            dataset=dataset,
            target_policy=target_policy,
            behavior_policy=behavior_policy,
            q_hat=fqe.q,
            v_hat=fqe.v,
            gamma=gamma,
        )
        return EstimatorResult(
            estimator_key="wdr",
            estimator_family="doubly_robust",
            estimate=float(np.mean(contributions)),
            native_diagnostic_kind=None,
            native_diagnostic_value=None,
            episode_contributions=contributions,
            auxiliary={},
            runtime_sec=perf_counter() - start,
        )

    if estimator_key == "dr_oracle":
        if truth_q is None or truth_v is None:
            raise ValueError("truth_q and truth_v are required for dr_oracle")
        contributions = doubly_robust_episode_contributions(
            dataset=dataset,
            target_policy=target_policy,
            behavior_policy=behavior_policy,
            q_hat=truth_q,
            v_hat=truth_v,
            gamma=gamma,
        )
        return EstimatorResult(
            estimator_key="dr_oracle",
            estimator_family="reference_model",
            estimate=float(np.mean(contributions)),
            native_diagnostic_kind=None,
            native_diagnostic_value=None,
            episode_contributions=contributions,
            auxiliary={},
            runtime_sec=perf_counter() - start,
        )

    raise ValueError(f"Unsupported estimator: {estimator_key}")


def estimator_registry() -> Dict[str, Callable[..., EstimatorResult]]:
    return {
        "is": evaluate_estimator,
        "wis": evaluate_estimator,
        "pdis": evaluate_estimator,
        "dr": evaluate_estimator,
        "wdr": evaluate_estimator,
        "fqe_linear": evaluate_estimator,
        "fqe_tabular": evaluate_estimator,
        "dr_oracle": evaluate_estimator,
    }
