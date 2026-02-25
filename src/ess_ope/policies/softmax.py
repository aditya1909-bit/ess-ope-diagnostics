from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ess_ope.policies.tabular import TabularPolicy


def stable_softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    z = logits / temperature
    z = z - np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)


def softmax_policy_from_logits(logits: np.ndarray, temperature: float = 1.0) -> TabularPolicy:
    return TabularPolicy(stable_softmax(logits, temperature=temperature))


def make_divergent_policy_pair(
    num_states: int,
    num_actions: int,
    alpha: float,
    seed: int,
    temperature: float = 1.0,
    target_logits: Optional[np.ndarray] = None,
) -> Tuple[TabularPolicy, TabularPolicy, np.ndarray, np.ndarray]:
    """Create target and behavior policies with a single divergence knob alpha."""
    rng = np.random.default_rng(seed)

    if target_logits is None:
        theta_pi = rng.normal(size=(num_states, num_actions))
    else:
        theta_pi = np.asarray(target_logits, dtype=float)

    theta_random = rng.normal(size=theta_pi.shape)
    theta_behavior = (1.0 - alpha) * theta_pi + alpha * theta_random

    target = softmax_policy_from_logits(theta_pi, temperature=temperature)
    behavior = softmax_policy_from_logits(theta_behavior, temperature=temperature)
    return target, behavior, theta_pi, theta_behavior


def statewise_kl(target: TabularPolicy, behavior: TabularPolicy, t: int = 0, eps: float = 1e-12) -> np.ndarray:
    t_probs = target.probs if target.probs.ndim == 2 else target.probs[t]
    b_probs = behavior.probs if behavior.probs.ndim == 2 else behavior.probs[t]

    b_safe = np.clip(b_probs, eps, 1.0)
    t_safe = np.clip(t_probs, eps, 1.0)
    return np.sum(t_safe * (np.log(t_safe) - np.log(b_safe)), axis=-1)
