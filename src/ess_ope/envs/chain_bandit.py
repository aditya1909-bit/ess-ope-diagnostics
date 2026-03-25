from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class ChainBanditConfig:
    num_states: int = 3
    num_actions: int = 5
    horizon: int = 10
    linear_feature_dim: int = 12
    transition_strength: float = 0.5
    reward_mean_scale: float = 1.0
    reward_gap: float = 0.5
    reward_std: float = 0.5
    beta: float = 0.0
    variant: str = "transitional"
    seed: int = 0


def _softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=axis, keepdims=True)


def _build_linear_features(
    horizon: int,
    num_states: int,
    num_actions: int,
    feature_dim: int,
) -> np.ndarray:
    base_dim = 10
    dim = max(int(feature_dim), base_dim)
    features = np.zeros((horizon, num_states, num_actions, dim), dtype=float)

    for t in range(horizon):
        t_norm = 0.0 if horizon <= 1 else t / (horizon - 1)
        for s in range(num_states):
            s_norm = 0.0 if num_states <= 1 else s / (num_states - 1)
            for a in range(num_actions):
                a_norm = 0.0 if num_actions <= 1 else a / (num_actions - 1)
                phi = np.array(
                    [
                        1.0,
                        t_norm,
                        s_norm,
                        a_norm,
                        t_norm * s_norm,
                        t_norm * a_norm,
                        s_norm * a_norm,
                        np.sin((t + 1) * (a + 1)),
                        np.cos((s + 1) * (a + 1)),
                        np.sin((t + 1) * (s + 1) * (a + 1) / max(1, num_states * num_actions)),
                    ],
                    dtype=float,
                )
                if dim > base_dim:
                    extra = np.zeros(dim - base_dim, dtype=float)
                    for idx in range(extra.size):
                        scale = idx + 1
                        extra[idx] = np.sin(scale * (t + 1)) + np.cos(scale * (s + 1 + a + 1))
                    phi = np.concatenate([phi, extra])
                features[t, s, a] = phi[:feature_dim]

    return features


class ChainBanditEnv:
    """Layered tabular environment that interpolates between bandits and an MDP."""

    def __init__(
        self,
        transition_probs: np.ndarray,
        rewards: np.ndarray,
        reward_std: float,
        initial_state_dist: np.ndarray,
        linear_sa_features: np.ndarray,
        config: ChainBanditConfig,
    ) -> None:
        self.transition_probs = np.asarray(transition_probs, dtype=float)
        self.rewards = np.asarray(rewards, dtype=float)
        self.reward_std = float(reward_std)
        self.initial_state_dist = np.asarray(initial_state_dist, dtype=float)
        self.linear_sa_features = np.asarray(linear_sa_features, dtype=float)
        self.config = config

        self.horizon = int(config.horizon)
        self.num_states = int(config.num_states)
        self.num_actions = int(config.num_actions)
        self.variant = str(config.variant)
        self.env_id = (
            f"chain_bandit_{self.variant}"
            f"_S{self.num_states}_A{self.num_actions}_H{self.horizon}"
            f"_ts{config.transition_strength:.3f}"
            f"_ms{config.reward_mean_scale:.3f}"
            f"_gap{config.reward_gap:.3f}"
            f"_std{config.reward_std:.3f}"
            f"_beta{config.beta:.3f}"
            f"_seed{config.seed}"
        )

        self._rng = np.random.default_rng(config.seed)
        self._state = 0
        self._t = 0

    @classmethod
    def generate(cls, config: ChainBanditConfig) -> "ChainBanditEnv":
        if config.variant not in {"reward_only", "transitional"}:
            raise ValueError("ChainBanditConfig.variant must be 'reward_only' or 'transitional'")

        rng = np.random.default_rng(config.seed)

        base_features = _build_linear_features(
            horizon=config.horizon,
            num_states=config.num_states,
            num_actions=config.num_actions,
            feature_dim=config.linear_feature_dim,
        )

        state_bias = rng.normal(loc=0.0, scale=0.35, size=(config.horizon, config.num_states))
        preferred_action = rng.integers(0, config.num_actions, size=(config.horizon, config.num_states))
        residual_reward = rng.normal(
            loc=0.0,
            scale=0.35,
            size=(config.horizon, config.num_states, config.num_actions),
        )

        rewards = np.zeros((config.horizon, config.num_states, config.num_actions), dtype=float)
        for t in range(config.horizon):
            for s in range(config.num_states):
                action_bonus = np.full(config.num_actions, -0.5 * config.reward_gap, dtype=float)
                action_bonus[preferred_action[t, s]] = config.reward_gap
                rewards[t, s] = config.reward_mean_scale * (
                    state_bias[t, s] + action_bonus + config.beta * residual_reward[t, s]
                )

        transition_probs = np.zeros(
            (config.horizon, config.num_states, config.num_actions, config.num_states),
            dtype=float,
        )
        base_next_logits = rng.normal(
            loc=0.0,
            scale=0.45,
            size=(config.horizon, config.num_states, config.num_states),
        )
        preferred_next_state = rng.integers(
            0,
            config.num_states,
            size=(config.horizon, config.num_states, config.num_actions),
        )
        residual_transition = rng.normal(
            loc=0.0,
            scale=0.30,
            size=(config.horizon, config.num_states, config.num_actions, config.num_states),
        )

        for t in range(config.horizon):
            for s in range(config.num_states):
                shared_logits = base_next_logits[t, s]
                for a in range(config.num_actions):
                    logits = shared_logits.copy()
                    if config.variant == "transitional":
                        logits = logits + config.beta * residual_transition[t, s, a]
                        logits[preferred_next_state[t, s, a]] += config.transition_strength
                    else:
                        logits = logits + config.beta * residual_transition[t, s, 0]
                    transition_probs[t, s, a] = _softmax(logits, axis=-1)

        initial_logits = rng.normal(size=config.num_states)
        initial_state_dist = _softmax(initial_logits, axis=-1)

        return cls(
            transition_probs=transition_probs,
            rewards=rewards,
            reward_std=config.reward_std,
            initial_state_dist=initial_state_dist,
            linear_sa_features=base_features,
            config=config,
        )

    def reset(self, seed: Optional[int] = None) -> int:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        self._state = int(self._rng.choice(self.num_states, p=self.initial_state_dist))
        return self._state

    def step(self, action: int) -> Tuple[int, float, bool, Dict[str, float]]:
        if self._t >= self.horizon:
            raise RuntimeError("Episode already terminated; call reset() before step().")

        action = int(action)
        mean = float(self.rewards[self._t, self._state, action])
        reward = float(self._rng.normal(loc=mean, scale=self.reward_std))
        next_state = int(self._rng.choice(self.num_states, p=self.transition_probs[self._t, self._state, action]))

        self._state = next_state
        self._t += 1
        done = self._t >= self.horizon
        return next_state, reward, done, {"reward_mean": mean}
