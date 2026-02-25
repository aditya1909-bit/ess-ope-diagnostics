from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ess_ope.data.dataset import EpisodeDataset
from ess_ope.policies.tabular import TabularPolicy


@dataclass
class ModelBasedResult:
    value: float
    v: np.ndarray
    q: np.ndarray


@dataclass
class FQEResult:
    value: float
    v: np.ndarray
    q: np.ndarray
    model_type: str
    weights_by_time: List[np.ndarray]


def evaluate_q_under_policy(q: np.ndarray, policy: TabularPolicy, t: int) -> np.ndarray:
    pi = policy.probs if policy.probs.ndim == 2 else policy.probs[t]
    return np.sum(pi * q, axis=-1)


def _ridge_regression(
    x: np.ndarray,
    y: np.ndarray,
    l2_reg: float,
    sample_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    if sample_weights is not None:
        w = np.sqrt(np.asarray(sample_weights, dtype=float))
        xw = x * w[:, None]
        yw = y * w
    else:
        xw = x
        yw = y

    xtx = xw.T @ xw
    xty = xw.T @ yw
    reg = l2_reg * np.eye(xtx.shape[0])
    try:
        return np.linalg.solve(xtx + reg, xty)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(xtx + reg) @ xty


def _tabular_features(states: np.ndarray, actions: np.ndarray, num_states: int, num_actions: int) -> np.ndarray:
    n = len(states)
    x = np.zeros((n, num_states * num_actions), dtype=float)
    idx = states * num_actions + actions
    x[np.arange(n), idx] = 1.0
    return x


def _predict_tabular_q(weights: np.ndarray, num_states: int, num_actions: int) -> np.ndarray:
    return weights.reshape(num_states, num_actions)


def _linear_features(states: np.ndarray, actions: np.ndarray, feature_tensor: np.ndarray) -> np.ndarray:
    return feature_tensor[states, actions]


def direct_model_tabular(
    dataset: EpisodeDataset,
    target_policy: TabularPolicy,
    num_states: int,
    num_actions: int,
    horizon: int,
    initial_state_dist: np.ndarray,
    gamma: float = 1.0,
    pseudo_count: float = 1e-3,
) -> ModelBasedResult:
    counts = np.full((horizon, num_states, num_actions, num_states), pseudo_count, dtype=float)
    reward_sum = np.zeros((horizon, num_states, num_actions), dtype=float)
    visits = np.zeros((horizon, num_states, num_actions), dtype=float)

    for t in range(horizon):
        s_t, a_t, r_t, sp_t, _ = dataset.transitions_at_time(t)
        for s, a, r, sp in zip(s_t, a_t, r_t, sp_t):
            counts[t, s, a, sp] += 1.0
            reward_sum[t, s, a] += r
            visits[t, s, a] += 1.0

    transition_hat = counts / counts.sum(axis=-1, keepdims=True)
    reward_hat = reward_sum / np.maximum(visits, 1.0)

    v = np.zeros((horizon + 1, num_states), dtype=float)
    q = np.zeros((horizon, num_states, num_actions), dtype=float)

    for t in range(horizon - 1, -1, -1):
        continuation = np.einsum("sak,k->sa", transition_hat[t], v[t + 1])
        q[t] = reward_hat[t] + gamma * continuation
        v[t] = evaluate_q_under_policy(q[t], target_policy, t)

    value = float(np.dot(initial_state_dist, v[0]))
    return ModelBasedResult(value=value, v=v, q=q)


def fitted_q_evaluation(
    dataset: EpisodeDataset,
    target_policy: TabularPolicy,
    num_states: int,
    num_actions: int,
    horizon: int,
    initial_state_dist: np.ndarray,
    model_type: str = "tabular",
    feature_tensor: Optional[np.ndarray] = None,
    gamma: float = 1.0,
    l2_reg: float = 1e-4,
    regression_weights: Optional[np.ndarray] = None,
) -> FQEResult:
    if model_type not in {"tabular", "linear"}:
        raise ValueError("model_type must be 'tabular' or 'linear'")
    if model_type == "linear" and feature_tensor is None:
        raise ValueError("feature_tensor is required for linear FQE")

    q = np.zeros((horizon, num_states, num_actions), dtype=float)
    v = np.zeros((horizon + 1, num_states), dtype=float)
    fitted_weights: List[np.ndarray] = []

    all_states = np.repeat(np.arange(num_states), num_actions)
    all_actions = np.tile(np.arange(num_actions), num_states)

    if model_type == "tabular":
        x_all = _tabular_features(all_states, all_actions, num_states, num_actions)
    else:
        x_all = _linear_features(all_states, all_actions, feature_tensor)

    for t in range(horizon - 1, -1, -1):
        s_t, a_t, r_t, sp_t, done_t = dataset.transitions_at_time(t)

        if t == horizon - 1:
            v_next = np.zeros_like(r_t, dtype=float)
        else:
            pi_next = target_policy.probs if target_policy.probs.ndim == 2 else target_policy.probs[t + 1]
            v_next = np.sum(pi_next[sp_t] * q[t + 1, sp_t], axis=-1)

        y = r_t + gamma * (1.0 - done_t.astype(float)) * v_next

        if model_type == "tabular":
            x = _tabular_features(s_t, a_t, num_states, num_actions)
        else:
            x = _linear_features(s_t, a_t, feature_tensor)

        sw = None if regression_weights is None else regression_weights[:, t]
        w_t = _ridge_regression(x, y, l2_reg=l2_reg, sample_weights=sw)
        fitted_weights.append(w_t)

        q_t_flat = x_all @ w_t
        q[t] = q_t_flat.reshape(num_states, num_actions)
        v[t] = evaluate_q_under_policy(q[t], target_policy, t)

    value = float(np.dot(initial_state_dist, v[0]))
    fitted_weights.reverse()
    return FQEResult(value=value, v=v, q=q, model_type=model_type, weights_by_time=fitted_weights)
