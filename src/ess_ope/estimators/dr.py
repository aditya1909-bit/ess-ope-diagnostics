from __future__ import annotations

import numpy as np

from ess_ope.data.dataset import EpisodeDataset
from ess_ope.estimators import compute_importance_weights
from ess_ope.policies.tabular import TabularPolicy


def doubly_robust_estimate(
    dataset: EpisodeDataset,
    target_policy: TabularPolicy,
    behavior_policy: TabularPolicy,
    q_hat: np.ndarray,
    v_hat: np.ndarray,
    gamma: float = 1.0,
    min_prob: float = 1e-12,
) -> float:
    """Finite-horizon per-decision doubly robust estimate."""
    weights = compute_importance_weights(
        dataset=dataset,
        target_policy=target_policy,
        behavior_policy=behavior_policy,
        min_prob=min_prob,
    )

    k, h = dataset.num_episodes, dataset.horizon
    estimates = np.zeros(k, dtype=float)

    for ep in range(k):
        s0 = dataset.states[ep, 0]
        dr = v_hat[0, s0]
        for t in range(h):
            s = dataset.states[ep, t]
            a = dataset.actions[ep, t]
            r = dataset.rewards[ep, t]
            sp = dataset.next_states[ep, t]
            done = dataset.dones[ep, t]

            td = r + gamma * (1.0 - float(done)) * v_hat[t + 1, sp] - q_hat[t, s, a]
            dr += weights.partial_weights[ep, t] * td
        estimates[ep] = dr

    return float(np.mean(estimates))
