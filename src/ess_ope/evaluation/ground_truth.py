from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ess_ope.envs.base import DiscreteFiniteHorizonEnv
from ess_ope.policies.tabular import TabularPolicy


@dataclass
class GroundTruthResult:
    value: float
    v: np.ndarray
    q: np.ndarray


@dataclass
class MonteCarloValueResult:
    value: float
    stderr: float


def dynamic_programming_value(
    env: DiscreteFiniteHorizonEnv,
    policy: TabularPolicy,
    gamma: float = 1.0,
) -> GroundTruthResult:
    h, s, a = env.horizon, env.num_states, env.num_actions

    v = np.zeros((h + 1, s), dtype=float)
    q = np.zeros((h, s, a), dtype=float)

    for t in range(h - 1, -1, -1):
        continuation = np.einsum("sak,k->sa", env.transition_probs[t], v[t + 1])
        q[t] = env.rewards[t] + gamma * continuation
        pi = policy.probs if policy.probs.ndim == 2 else policy.probs[t]
        v[t] = np.sum(pi * q[t], axis=-1)

    value = float(np.dot(env.initial_state_dist, v[0]))
    return GroundTruthResult(value=value, v=v, q=q)


def monte_carlo_policy_value(
    env: DiscreteFiniteHorizonEnv,
    policy: TabularPolicy,
    num_episodes: int,
    seed: int = 0,
    gamma: float = 1.0,
) -> MonteCarloValueResult:
    rng = np.random.default_rng(seed)
    returns = np.zeros(num_episodes, dtype=float)

    for ep in range(num_episodes):
        state = env.reset(seed=int(rng.integers(0, 2**32 - 1)))
        g = 0.0
        discount = 1.0
        for t in range(env.horizon):
            action = policy.sample_action(state, t=t, rng=rng)
            state, reward, done, _ = env.step(action)
            g += discount * reward
            discount *= gamma
            if done:
                break
        returns[ep] = g

    return MonteCarloValueResult(
        value=float(np.mean(returns)),
        stderr=float(np.std(returns, ddof=1) / np.sqrt(max(1, num_episodes))),
    )
