from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Dict

import numpy as np

from ess_ope.data.dataset import EpisodeDataset
from ess_ope.estimators import compute_importance_weights, is_family_estimates
from ess_ope.estimators.dm_fqe import direct_model_tabular, fitted_q_evaluation
from ess_ope.estimators.dr import doubly_robust_episode_contributions
from ess_ope.metrics.ess import episode_ess, normalized_weights
from ess_ope.policies.tabular import TabularPolicy


DISPLAY_NAMES = {
    "is": "IS",
    "snis": "SNIS",
    "pdis": "PDIS",
    "dm": "DM",
    "dr": "DR",
    "mrdr": "MRDR",
    "fqe": "FQE",
}


@dataclass
class EstimatorResult:
    estimator_key: str
    estimator_family: str
    estimate: float
    shared_wess: float
    wess_native_applicable: bool
    episode_contributions: np.ndarray | None = None
    auxiliary: Dict[str, Any] = field(default_factory=dict)
    runtime_sec: float = 0.0


def _trajectory_returns(dataset: EpisodeDataset, gamma: float) -> np.ndarray:
    discounts = gamma ** np.arange(dataset.horizon)
    return np.sum(dataset.rewards * discounts[None, :], axis=1)


def _snis_contributions(dataset: EpisodeDataset, episode_weights: np.ndarray, gamma: float) -> np.ndarray:
    returns = _trajectory_returns(dataset, gamma=gamma)
    norm = normalized_weights(episode_weights)
    return float(dataset.num_episodes) * norm * returns


def _shared_wess_from_weights(episode_weights: np.ndarray) -> float:
    p = normalized_weights(episode_weights)
    denom = np.sum(p**2)
    if denom <= 0:
        return 0.0
    return float(1.0 / denom)


def evaluate_estimator(
    estimator_key: str,
    dataset: EpisodeDataset,
    target_policy: TabularPolicy,
    behavior_policy: TabularPolicy,
    env: Any,
    gamma: float = 1.0,
) -> EstimatorResult:
    start = perf_counter()
    weights = compute_importance_weights(dataset, target_policy, behavior_policy)
    shared_wess = _shared_wess_from_weights(weights.episode_weights)

    if estimator_key in {"is", "snis", "pdis"}:
        is_res = is_family_estimates(dataset, target_policy, behavior_policy, gamma=gamma)
        if estimator_key == "is":
            contributions = weights.episode_weights * _trajectory_returns(dataset, gamma)
            estimate = float(is_res["is_trajectory"])
        elif estimator_key == "snis":
            contributions = _snis_contributions(dataset, weights.episode_weights, gamma)
            estimate = float(is_res["wis_trajectory"])
        else:
            contributions = np.sum(weights.partial_weights * dataset.rewards * (gamma ** np.arange(dataset.horizon))[None, :], axis=1)
            estimate = float(is_res["is_pdis"])
        return EstimatorResult(
            estimator_key=estimator_key,
            estimator_family="is_like",
            estimate=estimate,
            shared_wess=shared_wess,
            wess_native_applicable=True,
            episode_contributions=contributions,
            auxiliary={"episode_ess": episode_ess(weights.episode_weights)},
            runtime_sec=perf_counter() - start,
        )

    if estimator_key == "dm":
        dm = direct_model_tabular(
            dataset=dataset,
            target_policy=target_policy,
            num_states=env.num_states,
            num_actions=env.num_actions,
            horizon=env.horizon,
            initial_state_dist=env.initial_state_dist,
            gamma=gamma,
        )
        return EstimatorResult(
            estimator_key="dm",
            estimator_family="model_based",
            estimate=float(dm.value),
            shared_wess=shared_wess,
            wess_native_applicable=False,
            episode_contributions=None,
            runtime_sec=perf_counter() - start,
        )

    if estimator_key == "dr":
        dm = direct_model_tabular(
            dataset=dataset,
            target_policy=target_policy,
            num_states=env.num_states,
            num_actions=env.num_actions,
            horizon=env.horizon,
            initial_state_dist=env.initial_state_dist,
            gamma=gamma,
        )
        contributions = doubly_robust_episode_contributions(
            dataset=dataset,
            target_policy=target_policy,
            behavior_policy=behavior_policy,
            q_hat=dm.q,
            v_hat=dm.v,
            gamma=gamma,
        )
        return EstimatorResult(
            estimator_key="dr",
            estimator_family="doubly_robust",
            estimate=float(np.mean(contributions)),
            shared_wess=shared_wess,
            wess_native_applicable=False,
            episode_contributions=contributions,
            runtime_sec=perf_counter() - start,
        )

    if estimator_key == "mrdr":
        mrdr = fitted_q_evaluation(
            dataset=dataset,
            target_policy=target_policy,
            num_states=env.num_states,
            num_actions=env.num_actions,
            horizon=env.horizon,
            initial_state_dist=env.initial_state_dist,
            model_type="tabular",
            gamma=gamma,
            l2_reg=1e-5,
            regression_weights=weights.partial_weights,
        )
        contributions = doubly_robust_episode_contributions(
            dataset=dataset,
            target_policy=target_policy,
            behavior_policy=behavior_policy,
            q_hat=mrdr.q,
            v_hat=mrdr.v,
            gamma=gamma,
        )
        return EstimatorResult(
            estimator_key="mrdr",
            estimator_family="doubly_robust",
            estimate=float(np.mean(contributions)),
            shared_wess=shared_wess,
            wess_native_applicable=False,
            episode_contributions=contributions,
            runtime_sec=perf_counter() - start,
        )

    if estimator_key == "fqe":
        fqe = fitted_q_evaluation(
            dataset=dataset,
            target_policy=target_policy,
            num_states=env.num_states,
            num_actions=env.num_actions,
            horizon=env.horizon,
            initial_state_dist=env.initial_state_dist,
            model_type="tabular",
            gamma=gamma,
            l2_reg=1e-5,
        )
        return EstimatorResult(
            estimator_key="fqe",
            estimator_family="value_model",
            estimate=float(fqe.value),
            shared_wess=shared_wess,
            wess_native_applicable=False,
            episode_contributions=None,
            runtime_sec=perf_counter() - start,
        )

    raise ValueError(f"Unsupported estimator: {estimator_key}")
