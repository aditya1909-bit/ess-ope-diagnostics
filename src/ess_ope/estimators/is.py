from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from ess_ope.data.dataset import EpisodeDataset
from ess_ope.policies.tabular import TabularPolicy


@dataclass
class ISWeightResult:
    ratios: np.ndarray
    partial_weights: np.ndarray
    episode_weights: np.ndarray


def _policy_prob_for_batch(policy: TabularPolicy, states: np.ndarray, actions: np.ndarray, t: int) -> np.ndarray:
    if policy.probs.ndim == 2:
        probs = policy.probs[states, actions]
    else:
        probs = policy.probs[t, states, actions]
    return np.asarray(probs, dtype=float)


def compute_importance_weights(
    dataset: EpisodeDataset,
    target_policy: TabularPolicy,
    behavior_policy: TabularPolicy,
    min_prob: float = 1e-12,
    clip_ratio: Optional[float] = None,
) -> ISWeightResult:
    k, h = dataset.num_episodes, dataset.horizon
    ratios = np.ones((k, h), dtype=float)

    for t in range(h):
        states = dataset.states[:, t]
        actions = dataset.actions[:, t]

        pi = _policy_prob_for_batch(target_policy, states, actions, t)
        mu = _policy_prob_for_batch(behavior_policy, states, actions, t)
        ratio = pi / np.maximum(mu, min_prob)

        if clip_ratio is not None:
            ratio = np.minimum(ratio, clip_ratio)

        ratios[:, t] = ratio

    partial = np.cumprod(ratios, axis=1)
    partial = np.clip(partial, 0.0, 1e300)
    episode = partial[:, -1]
    return ISWeightResult(ratios=ratios, partial_weights=partial, episode_weights=episode)


def trajectory_is(dataset: EpisodeDataset, episode_weights: np.ndarray, gamma: float = 1.0) -> float:
    contributions = trajectory_is_episode_contributions(dataset, episode_weights, gamma=gamma)
    return float(np.mean(contributions))


def weighted_trajectory_is(dataset: EpisodeDataset, episode_weights: np.ndarray, gamma: float = 1.0) -> float:
    discounts = gamma ** np.arange(dataset.horizon)
    returns = np.sum(dataset.rewards * discounts[None, :], axis=1)
    denom = np.sum(episode_weights)
    if denom <= 0:
        return float("nan")
    return float(np.sum(episode_weights * returns) / denom)


def per_decision_is(dataset: EpisodeDataset, partial_weights: np.ndarray, gamma: float = 1.0) -> float:
    contributions = per_decision_is_episode_contributions(dataset, partial_weights, gamma=gamma)
    return float(np.mean(contributions))


def weighted_per_decision_is(dataset: EpisodeDataset, partial_weights: np.ndarray, gamma: float = 1.0) -> float:
    discounts = gamma ** np.arange(dataset.horizon)
    estimate = 0.0
    for t in range(dataset.horizon):
        w_t = partial_weights[:, t]
        denom = np.sum(w_t)
        if denom <= 0:
            continue
        estimate += discounts[t] * float(np.sum(w_t * dataset.rewards[:, t]) / denom)
    return estimate


def trajectory_is_episode_contributions(
    dataset: EpisodeDataset,
    episode_weights: np.ndarray,
    gamma: float = 1.0,
) -> np.ndarray:
    discounts = gamma ** np.arange(dataset.horizon)
    returns = np.sum(dataset.rewards * discounts[None, :], axis=1)
    return np.asarray(episode_weights, dtype=float) * returns


def per_decision_is_episode_contributions(
    dataset: EpisodeDataset,
    partial_weights: np.ndarray,
    gamma: float = 1.0,
) -> np.ndarray:
    discounts = gamma ** np.arange(dataset.horizon)
    return np.sum(partial_weights * dataset.rewards * discounts[None, :], axis=1)


def is_family_estimates(
    dataset: EpisodeDataset,
    target_policy: TabularPolicy,
    behavior_policy: TabularPolicy,
    gamma: float = 1.0,
    min_prob: float = 1e-12,
    clip_ratio: Optional[float] = None,
) -> Dict[str, np.ndarray | float]:
    weights = compute_importance_weights(
        dataset=dataset,
        target_policy=target_policy,
        behavior_policy=behavior_policy,
        min_prob=min_prob,
        clip_ratio=clip_ratio,
    )

    return {
        "is_trajectory": trajectory_is(dataset, weights.episode_weights, gamma=gamma),
        "wis_trajectory": weighted_trajectory_is(dataset, weights.episode_weights, gamma=gamma),
        "is_pdis": per_decision_is(dataset, weights.partial_weights, gamma=gamma),
        "wis_pdis": weighted_per_decision_is(dataset, weights.partial_weights, gamma=gamma),
        "episode_weights": weights.episode_weights,
        "partial_weights": weights.partial_weights,
    }
