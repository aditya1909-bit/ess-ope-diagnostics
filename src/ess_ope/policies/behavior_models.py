from __future__ import annotations

import numpy as np

from ess_ope.policies.tabular import TabularPolicy


def mixture_behavior_policy(target_policy: TabularPolicy, mix: float) -> TabularPolicy:
    """Blend target policy with uniform to improve support."""
    if not 0.0 <= mix <= 1.0:
        raise ValueError("mix must be in [0, 1]")

    probs = target_policy.probs
    num_actions = probs.shape[-1]
    uniform = np.full_like(probs, 1.0 / num_actions)
    mixed = (1.0 - mix) * probs + mix * uniform
    mixed = mixed / mixed.sum(axis=-1, keepdims=True)
    return TabularPolicy(mixed)


def epsilon_greedy_from_q(q_values: np.ndarray, epsilon: float) -> TabularPolicy:
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be in [0, 1]")

    if q_values.ndim != 2:
        raise ValueError("q_values must have shape (S, A)")

    s, a = q_values.shape
    probs = np.full((s, a), epsilon / a)
    greedy_actions = np.argmax(q_values, axis=1)
    probs[np.arange(s), greedy_actions] += 1.0 - epsilon
    probs = probs / probs.sum(axis=-1, keepdims=True)
    return TabularPolicy(probs)
