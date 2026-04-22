from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from ess_ope.envs.base import DiscreteFiniteHorizonEnv
from ess_ope.policies.tabular import TabularPolicy


MISMATCH_MIX = {"low": 0.15, "medium": 0.35, "high": 0.60}
BEHAVIOR_FLOOR = 0.02
REWARD_VARIANCE_SCALE = {"low": 0.15, "medium": 0.45, "high": 0.90}


@dataclass
class StudyEnvironmentBundle:
    env: StochasticTabularEnv
    target_policy: TabularPolicy
    behavior_policy: TabularPolicy
    metadata: Dict[str, Any]


class StochasticTabularEnv(DiscreteFiniteHorizonEnv):
    """Tabular finite-horizon environment with stochastic Gaussian rewards."""

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
            raise ValueError("reward_stds must match rewards shape")
        self.reward_stds = reward_stds
        self.linear_sa_features = np.asarray(linear_sa_features, dtype=float)
        self.env_id = env_id

    def step(self, action: int):
        next_state, _, done, info = super().step(action)
        mean = float(self.rewards[self._t - 1, info["prev_state"], action])
        std = float(max(0.0, self.reward_stds[self._t - 1, info["prev_state"], action]))
        reward = float(self._rng.normal(loc=mean, scale=std)) if std > 0 else mean
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


def _softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = np.asarray(logits, dtype=float) / max(temperature, 1e-8)
    shifted = scaled - np.max(scaled, axis=-1, keepdims=True)
    probs = np.exp(shifted)
    probs /= np.sum(probs, axis=-1, keepdims=True)
    return probs


def _apply_floor(probs: np.ndarray, floor: float) -> np.ndarray:
    clipped = np.maximum(np.asarray(probs, dtype=float), float(floor))
    clipped /= np.sum(clipped, axis=-1, keepdims=True)
    return clipped


def _anti_policy_probs(target_probs: np.ndarray) -> np.ndarray:
    anti = np.max(target_probs, axis=-1, keepdims=True) - target_probs + 1e-3
    anti /= np.sum(anti, axis=-1, keepdims=True)
    return anti


def _build_policy_pair(policy_logits: np.ndarray, mismatch_level: str) -> tuple[TabularPolicy, TabularPolicy]:
    target_probs = _softmax(policy_logits, temperature=0.85)
    anti_probs = _anti_policy_probs(target_probs)
    mix = float(MISMATCH_MIX[str(mismatch_level)])
    behavior_probs = (1.0 - mix) * target_probs + mix * anti_probs
    behavior_probs = _apply_floor(behavior_probs, BEHAVIOR_FLOOR)
    target_probs = _apply_floor(target_probs, 1e-4)
    return TabularPolicy(target_probs), TabularPolicy(behavior_probs)


def _bandit_reward_means(rng: np.random.Generator, num_states: int, num_actions: int, structure: str) -> np.ndarray:
    raise RuntimeError("_bandit_reward_means requires target and behavior policies; use _bandit_reward_means_with_policies instead")


def _center_under_target(values: np.ndarray, target_probs: np.ndarray) -> np.ndarray:
    centered = np.asarray(values, dtype=float).copy()
    centered -= np.sum(centered * target_probs, axis=-1, keepdims=True)
    return centered


def _bandit_reward_means_with_policies(
    rng: np.random.Generator,
    target_probs: np.ndarray,
    behavior_probs: np.ndarray,
    structure: str,
) -> np.ndarray:
    num_states, num_actions = target_probs.shape
    states = np.linspace(-1.0, 1.0, num_states)
    base_state = 0.55 * np.sin(1.8 * states) + 0.35 * states
    action_offsets = np.linspace(-0.5, 0.5, num_actions)

    if structure == "smooth":
        smooth_shape = np.tile(action_offsets[None, :], (num_states, 1))
        deviations = 0.12 * _center_under_target(smooth_shape, target_probs)
        return (base_state[:, None] + deviations)[None, :, :]

    if structure == "heterogeneous":
        ratios = target_probs / np.maximum(behavior_probs, 1e-12)
        favored_actions = np.argmax(ratios, axis=1)
        spike_shape = np.zeros((num_states, num_actions), dtype=float)
        spike_shape[np.arange(num_states), favored_actions] = 1.0
        spike_shape += 0.10 * rng.normal(size=(num_states, num_actions))
        deviations = 1.75 * _center_under_target(spike_shape, target_probs)
        return (base_state[:, None] + deviations)[None, :, :]

    raise ValueError(f"Unsupported reward_mean_structure: {structure}")


def build_contextual_bandit(seed: int, mismatch_level: str, reward_variance_regime: str, reward_mean_structure: str) -> StudyEnvironmentBundle:
    rng = np.random.default_rng(seed)
    num_states = 20
    num_actions = 4
    horizon = 1

    state_effect = rng.normal(scale=0.45, size=(num_states, 1))
    action_effect = rng.normal(scale=0.6, size=(1, num_actions))
    policy_logits = state_effect + action_effect + 0.25 * rng.normal(size=(num_states, num_actions))
    target_policy, behavior_policy = _build_policy_pair(policy_logits, mismatch_level)

    rewards = _bandit_reward_means_with_policies(
        rng,
        target_probs=target_policy.probs,
        behavior_probs=behavior_policy.probs,
        structure=reward_mean_structure,
    )
    reward_stds = np.full_like(rewards, REWARD_VARIANCE_SCALE[str(reward_variance_regime)])
    transition_probs = np.zeros((1, num_states, num_actions, num_states), dtype=float)
    for s in range(num_states):
        transition_probs[0, s, :, s] = 1.0
    initial_state_dist = np.full(num_states, 1.0 / num_states, dtype=float)
    env = StochasticTabularEnv(
        transition_probs=transition_probs,
        rewards=rewards,
        reward_stds=reward_stds,
        initial_state_dist=initial_state_dist,
        horizon=horizon,
        linear_sa_features=build_linear_features(horizon, num_states, num_actions, 6),
        env_id=f"bandit_seed{seed}_{reward_variance_regime}_{reward_mean_structure}",
        seed=seed,
    )
    return StudyEnvironmentBundle(
        env=env,
        target_policy=target_policy,
        behavior_policy=behavior_policy,
        metadata={
            "env_name": "contextual_bandit",
            "reward_variance_regime": reward_variance_regime,
            "reward_mean_structure": reward_mean_structure,
        },
    )


def _random_branching_transitions(
    rng: np.random.Generator,
    horizon: int,
    num_states: int,
    num_actions: int,
    branch_factor: int,
    self_bias: float,
) -> np.ndarray:
    transition = np.zeros((horizon, num_states, num_actions, num_states), dtype=float)
    for t in range(horizon):
        for s in range(num_states):
            for a in range(num_actions):
                support = rng.choice(num_states, size=min(branch_factor, num_states), replace=False)
                weights = rng.uniform(0.1, 1.0, size=support.size)
                if s not in support:
                    support[0] = s
                weights[np.where(support == s)[0][0]] += self_bias
                transition[t, s, a, support] = weights / np.sum(weights)
    return transition


def _dense_rewards(rng: np.random.Generator, horizon: int, num_states: int, num_actions: int) -> np.ndarray:
    t_grid = np.linspace(0.0, 1.0, horizon)[:, None, None]
    s_grid = np.linspace(-1.0, 1.0, num_states)[None, :, None]
    a_grid = np.linspace(-0.8, 0.8, num_actions)[None, None, :]
    rewards = 0.65 * np.sin(2.0 * np.pi * (s_grid + 0.15 * t_grid)) + 0.35 * a_grid
    rewards += 0.15 * rng.normal(size=(horizon, num_states, num_actions))
    return rewards


def _sparse_terminal_rewards(
    rng: np.random.Generator,
    horizon: int,
    num_states: int,
    num_actions: int,
    scale: float,
) -> np.ndarray:
    rewards = np.zeros((horizon, num_states, num_actions), dtype=float)
    goal_states = rng.choice(num_states, size=max(2, num_states // 8), replace=False)
    preferred_actions = rng.integers(0, num_actions, size=goal_states.size)
    final_t = horizon - 1
    for idx, state in enumerate(goal_states):
        rewards[final_t, state, preferred_actions[idx]] = scale
        rewards[final_t, state] += 0.1 * scale
    if horizon >= 3:
        rewards[-2] += 0.2 * rewards[-1]
    return rewards


def _time_varying_policy_logits(rng: np.random.Generator, horizon: int, num_states: int, num_actions: int) -> np.ndarray:
    base = rng.normal(scale=0.6, size=(horizon, num_states, num_actions))
    time_bias = np.linspace(-0.4, 0.4, horizon)[:, None, None]
    action_bias = np.linspace(-0.6, 0.6, num_actions)[None, None, :]
    state_bias = rng.normal(scale=0.3, size=(1, num_states, 1))
    return base + time_bias + action_bias + state_bias


def build_short_horizon_mdp(seed: int, mismatch_level: str, reward_variant: str) -> StudyEnvironmentBundle:
    rng = np.random.default_rng(seed)
    horizon = 5
    num_states = 30
    num_actions = 4
    transition_probs = _random_branching_transitions(rng, horizon, num_states, num_actions, branch_factor=4, self_bias=1.1)
    rewards = _dense_rewards(rng, horizon, num_states, num_actions) if reward_variant == "dense" else _sparse_terminal_rewards(rng, horizon, num_states, num_actions, scale=4.5)
    reward_stds = np.full_like(rewards, 0.25 if reward_variant == "dense" else 0.15)
    policy_logits = _time_varying_policy_logits(rng, horizon, num_states, num_actions)
    target_policy, behavior_policy = _build_policy_pair(policy_logits, mismatch_level)
    env = StochasticTabularEnv(
        transition_probs=transition_probs,
        rewards=rewards,
        reward_stds=reward_stds,
        initial_state_dist=np.full(num_states, 1.0 / num_states, dtype=float),
        horizon=horizon,
        linear_sa_features=build_linear_features(horizon, num_states, num_actions, 8),
        env_id=f"short_mdp_seed{seed}_{reward_variant}",
        seed=seed,
    )
    return StudyEnvironmentBundle(
        env=env,
        target_policy=target_policy,
        behavior_policy=behavior_policy,
        metadata={"env_name": "short_tabular_mdp", "reward_variant": reward_variant},
    )


def build_long_horizon_mdp(seed: int, mismatch_level: str, reward_variant: str = "sparse_late") -> StudyEnvironmentBundle:
    rng = np.random.default_rng(seed)
    horizon = 20
    num_states = 50
    num_actions = 4
    transition_probs = _random_branching_transitions(rng, horizon, num_states, num_actions, branch_factor=5, self_bias=0.7)
    rewards = _sparse_terminal_rewards(rng, horizon, num_states, num_actions, scale=7.0)
    rewards += 0.10 * _dense_rewards(rng, horizon, num_states, num_actions)
    reward_stds = np.full_like(rewards, 0.20)
    reward_stds[-3:] *= 0.8
    policy_logits = _time_varying_policy_logits(rng, horizon, num_states, num_actions)
    target_policy, behavior_policy = _build_policy_pair(policy_logits, mismatch_level)
    env = StochasticTabularEnv(
        transition_probs=transition_probs,
        rewards=rewards,
        reward_stds=reward_stds,
        initial_state_dist=np.full(num_states, 1.0 / num_states, dtype=float),
        horizon=horizon,
        linear_sa_features=build_linear_features(horizon, num_states, num_actions, 8),
        env_id=f"long_mdp_seed{seed}_{reward_variant}",
        seed=seed,
    )
    return StudyEnvironmentBundle(
        env=env,
        target_policy=target_policy,
        behavior_policy=behavior_policy,
        metadata={"env_name": "long_tabular_mdp", "reward_variant": reward_variant},
    )


def build_long_stress_mdp(seed: int) -> StudyEnvironmentBundle:
    rng = np.random.default_rng(seed)
    horizon = 12
    num_states = 50
    num_actions = 4
    transition_probs = _random_branching_transitions(rng, horizon, num_states, num_actions, branch_factor=4, self_bias=0.9)
    rewards = 0.45 * _dense_rewards(rng, horizon, num_states, num_actions)
    rewards += _sparse_terminal_rewards(rng, horizon, num_states, num_actions, scale=5.0)
    reward_stds = np.full_like(rewards, 0.15)
    policy_logits = _time_varying_policy_logits(rng, horizon, num_states, num_actions)
    target_policy, behavior_policy = _build_policy_pair(policy_logits, mismatch_level="low")
    env = StochasticTabularEnv(
        transition_probs=transition_probs,
        rewards=rewards,
        reward_stds=reward_stds,
        initial_state_dist=np.full(num_states, 1.0 / num_states, dtype=float),
        horizon=horizon,
        linear_sa_features=build_linear_features(horizon, num_states, num_actions, 8),
        env_id=f"long_stress_seed{seed}",
        seed=seed,
    )
    return StudyEnvironmentBundle(
        env=env,
        target_policy=target_policy,
        behavior_policy=behavior_policy,
        metadata={"env_name": "long_tabular_mdp", "reward_variant": "long_stress", "calibration_role": "stress"},
    )


def build_environment_bundle(environment: Dict[str, Any], seed: int, condition: Dict[str, Any]) -> StudyEnvironmentBundle:
    name = str(environment.get("name", "contextual_bandit"))
    mismatch_level = str(condition.get("mismatch_level", environment.get("mismatch_level", "medium")))
    reward_variance_regime = str(condition.get("reward_variance_regime", environment.get("reward_variance_regime", "medium")))
    reward_mean_structure = str(condition.get("reward_mean_structure", environment.get("reward_mean_structure", "smooth")))
    reward_variant = str(condition.get("reward_variant", environment.get("reward_variant", "dense")))
    calibration_env = str(condition.get("calibration_env", "short"))

    if name == "contextual_bandit":
        return build_contextual_bandit(
            seed=seed,
            mismatch_level=mismatch_level,
            reward_variance_regime=reward_variance_regime,
            reward_mean_structure=reward_mean_structure,
        )
    if name == "short_tabular_mdp":
        return build_short_horizon_mdp(seed=seed, mismatch_level=mismatch_level, reward_variant=reward_variant)
    if name == "long_tabular_mdp":
        return build_long_horizon_mdp(seed=seed, mismatch_level=mismatch_level, reward_variant=reward_variant)
    if name == "fqe_calibration_pair":
        if calibration_env == "long_stress":
            return build_long_stress_mdp(seed=seed)
        bundle = build_short_horizon_mdp(seed=seed, mismatch_level="low", reward_variant="dense")
        bundle.metadata["calibration_role"] = "positive"
        return bundle
    raise ValueError(f"Unsupported environment name: {name}")
