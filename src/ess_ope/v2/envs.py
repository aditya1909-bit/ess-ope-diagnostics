from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from ess_ope.envs.base import DiscreteFiniteHorizonEnv


@dataclass
class EnvironmentBundle:
    env: "StochasticTabularEnv"
    metadata: Dict[str, float | str | int]


class StochasticTabularEnv(DiscreteFiniteHorizonEnv):
    """Tabular finite-horizon environment with stochastic rewards around known means."""

    def __init__(
        self,
        transition_probs: np.ndarray,
        rewards: np.ndarray,
        reward_stds: np.ndarray,
        initial_state_dist: np.ndarray,
        horizon: int,
        linear_sa_features: np.ndarray,
        env_id: str,
        seed: int = 0,
        rare_state_mask: np.ndarray | None = None,
    ) -> None:
        super().__init__(
            transition_probs=transition_probs,
            rewards=rewards,
            initial_state_dist=initial_state_dist,
            horizon=horizon,
            seed=seed,
        )
        reward_stds = np.asarray(reward_stds, dtype=float)
        if reward_stds.shape != rewards.shape:
            raise ValueError("reward_stds must match reward means shape")
        self.reward_stds = reward_stds
        self.linear_sa_features = np.asarray(linear_sa_features, dtype=float)
        self.env_id = env_id
        self.rare_state_mask = (
            np.asarray(rare_state_mask, dtype=bool)
            if rare_state_mask is not None
            else np.zeros(self.num_states, dtype=bool)
        )

    def step(self, action: int):
        if self._done:
            raise RuntimeError("Episode already finished; call reset().")
        if not 0 <= action < self.num_actions:
            raise ValueError("Action out of bounds")

        probs = self.transition_probs[self._t, self._state, action]
        next_state = int(self._rng.choice(self.num_states, p=probs))
        mean = float(self.rewards[self._t, self._state, action])
        std = float(max(0.0, self.reward_stds[self._t, self._state, action]))
        reward = float(self._rng.normal(loc=mean, scale=std)) if std > 0 else mean

        self._t += 1
        done = self._t >= self.horizon
        self._done = done
        prev_state = self._state
        self._state = next_state
        info = {"t": self._t, "prev_state": prev_state, "reward_mean": mean, "reward_std": std}
        return next_state, reward, done, info


def build_linear_features(horizon: int, num_states: int, num_actions: int, feature_dim: int) -> np.ndarray:
    feature_dim = max(6, int(feature_dim))
    feats = np.zeros((horizon, num_states, num_actions, feature_dim), dtype=float)
    for t in range(horizon):
        t_norm = 0.0 if horizon <= 1 else t / (horizon - 1)
        for s in range(num_states):
            s_norm = 0.0 if num_states <= 1 else s / (num_states - 1)
            for a in range(num_actions):
                a_norm = 0.0 if num_actions <= 1 else a / (num_actions - 1)
                base = np.array(
                    [
                        1.0,
                        t_norm,
                        s_norm,
                        a_norm,
                        s_norm * a_norm,
                        (t_norm + 1.0) * (a_norm + 0.1),
                    ],
                    dtype=float,
                )
                if feature_dim > len(base):
                    extra = np.zeros(feature_dim - len(base), dtype=float)
                    for idx in range(extra.size):
                        extra[idx] = np.sin((idx + 1) * (t + 1)) + np.cos((idx + 1) * (s + a + 1))
                    base = np.concatenate([base, extra])
                feats[t, s, a] = base[:feature_dim]
    return feats


def _reward_noise_scale(level: str) -> float:
    mapping = {"low": 0.1, "medium": 0.35, "high": 0.7}
    return float(mapping.get(str(level), 0.35))


def _rarity_probability(level: str) -> float:
    mapping = {"none": 0.25, "low": 0.15, "medium": 0.06, "high": 0.02}
    return float(mapping.get(str(level), 0.06))


def build_contextual_bandit(
    seed: int,
    sample_size: int,
    reward_noise_level: str,
    env_cfg: Dict[str, float | int],
) -> EnvironmentBundle:
    rng = np.random.default_rng(seed)
    num_states = int(env_cfg.get("num_contexts", env_cfg.get("num_states", 6)))
    num_actions = int(env_cfg.get("num_actions", 3))
    feature_dim = int(env_cfg.get("linear_feature_dim", 8))
    action_gap = float(env_cfg.get("action_gap", 0.6))

    base = rng.normal(scale=0.4, size=(num_states,))
    preferred_actions = rng.integers(0, num_actions, size=num_states)
    rewards = np.zeros((1, num_states, num_actions), dtype=float)
    reward_stds = np.zeros_like(rewards)
    for s in range(num_states):
        action_effect = np.full(num_actions, -0.5 * action_gap, dtype=float)
        action_effect[preferred_actions[s]] = action_gap
        heterosk = np.linspace(0.8, 1.2, num_actions)
        rewards[0, s] = base[s] + action_effect
        reward_stds[0, s] = _reward_noise_scale(reward_noise_level) * heterosk

    transition = np.zeros((1, num_states, num_actions, num_states), dtype=float)
    for s in range(num_states):
        transition[0, s, :, s] = 1.0
    initial = rng.dirichlet(np.ones(num_states))
    linear_features = build_linear_features(1, num_states, num_actions, feature_dim)
    env = StochasticTabularEnv(
        transition_probs=transition,
        rewards=rewards,
        reward_stds=reward_stds,
        initial_state_dist=initial,
        horizon=1,
        linear_sa_features=linear_features,
        env_id=f"contextual_bandit_s{num_states}_a{num_actions}_seed{seed}_n{sample_size}",
        seed=seed,
    )
    return EnvironmentBundle(env=env, metadata={"action_gap": action_gap, "num_states": num_states, "num_actions": num_actions})


def build_tabular_mdp(
    seed: int,
    horizon: int,
    reward_noise_level: str,
    env_cfg: Dict[str, float | int],
) -> EnvironmentBundle:
    rng = np.random.default_rng(seed)
    num_states = int(env_cfg.get("num_states", 18))
    num_actions = int(env_cfg.get("num_actions", 3))
    feature_dim = int(env_cfg.get("linear_feature_dim", 10))
    branch_factor = int(env_cfg.get("branch_factor", min(4, num_states)))
    late_reward_multiplier = float(env_cfg.get("late_reward_multiplier", 1.5))

    transition = np.zeros((horizon, num_states, num_actions, num_states), dtype=float)
    for t in range(horizon):
        for s in range(num_states):
            for a in range(num_actions):
                support = rng.choice(num_states, size=min(branch_factor, num_states), replace=False)
                probs = rng.dirichlet(np.ones(len(support)))
                transition[t, s, a, support] = probs

    linear_features = build_linear_features(horizon, num_states, num_actions, feature_dim)
    feature_signal = linear_features[..., 1] + 0.75 * linear_features[..., 4]
    rewards = np.tanh(feature_signal + rng.normal(scale=0.35, size=(horizon, num_states, num_actions)))
    time_scale = np.linspace(1.0, late_reward_multiplier, horizon)[:, None, None]
    rewards = rewards * time_scale
    reward_stds = np.full_like(rewards, _reward_noise_scale(reward_noise_level))
    reward_stds *= np.linspace(0.6, late_reward_multiplier, horizon)[:, None, None]
    initial = rng.dirichlet(np.ones(num_states))
    env = StochasticTabularEnv(
        transition_probs=transition,
        rewards=rewards,
        reward_stds=reward_stds,
        initial_state_dist=initial,
        horizon=horizon,
        linear_sa_features=linear_features,
        env_id=f"tabular_mdp_s{num_states}_a{num_actions}_h{horizon}_seed{seed}",
        seed=seed,
    )
    return EnvironmentBundle(env=env, metadata={"branch_factor": branch_factor, "late_reward_multiplier": late_reward_multiplier})


def build_rare_event_mdp(
    seed: int,
    horizon: int,
    rarity_level: str,
    reward_noise_level: str,
    env_cfg: Dict[str, float | int],
) -> EnvironmentBundle:
    rng = np.random.default_rng(seed)
    num_states = int(env_cfg.get("num_states", 20))
    num_actions = int(env_cfg.get("num_actions", 3))
    feature_dim = int(env_cfg.get("linear_feature_dim", 10))
    rare_reward = float(env_cfg.get("rare_reward", 12.0))
    rare_event_prob = _rarity_probability(rarity_level)

    rare_states = max(1, int(round(num_states * rare_event_prob)))
    rare_mask = np.zeros(num_states, dtype=bool)
    rare_idx = rng.choice(num_states, size=rare_states, replace=False)
    rare_mask[rare_idx] = True

    transition = np.zeros((horizon, num_states, num_actions, num_states), dtype=float)
    safe_state = int(rng.integers(0, num_states))
    for t in range(horizon):
        for s in range(num_states):
            for a in range(num_actions):
                base = np.full(num_states, 1e-8, dtype=float)
                base[safe_state] += 1.0
                if a == (t + s) % num_actions:
                    base[rare_mask] += 1.2
                else:
                    base[rare_mask] += 0.2
                if rare_mask[s]:
                    base[rare_mask] += 0.6
                transition[t, s, a] = base / np.sum(base)

    rewards = np.zeros((horizon, num_states, num_actions), dtype=float)
    rewards[:, rare_mask, :] = rare_reward
    reward_stds = np.full_like(rewards, _reward_noise_scale(reward_noise_level))
    reward_stds[:, rare_mask, :] *= 0.5
    linear_features = build_linear_features(horizon, num_states, num_actions, feature_dim)
    initial = np.full(num_states, (1.0 - rare_event_prob) / max(1, num_states - rare_states))
    initial[rare_mask] = rare_event_prob / rare_states
    initial = initial / np.sum(initial)
    env = StochasticTabularEnv(
        transition_probs=transition,
        rewards=rewards,
        reward_stds=reward_stds,
        initial_state_dist=initial,
        horizon=horizon,
        linear_sa_features=linear_features,
        env_id=f"rare_event_mdp_s{num_states}_a{num_actions}_h{horizon}_rarity{rarity_level}_seed{seed}",
        seed=seed,
        rare_state_mask=rare_mask,
    )
    return EnvironmentBundle(
        env=env,
        metadata={"rare_states": int(rare_states), "rare_event_prob": float(rare_event_prob), "rare_reward": rare_reward},
    )


def build_environment(
    env_family: str,
    seed: int,
    sample_size: int,
    horizon: int,
    reward_noise_level: str,
    rarity_level: str,
    env_cfg: Dict[str, float | int],
) -> EnvironmentBundle:
    family = str(env_family)
    if family == "contextual_bandit":
        return build_contextual_bandit(seed=seed, sample_size=sample_size, reward_noise_level=reward_noise_level, env_cfg=env_cfg)
    if family == "tabular_mdp":
        return build_tabular_mdp(seed=seed, horizon=horizon, reward_noise_level=reward_noise_level, env_cfg=env_cfg)
    if family == "rare_event_mdp":
        return build_rare_event_mdp(
            seed=seed,
            horizon=horizon,
            rarity_level=rarity_level,
            reward_noise_level=reward_noise_level,
            env_cfg=env_cfg,
        )
    raise ValueError(f"Unsupported env_family: {env_family}")
