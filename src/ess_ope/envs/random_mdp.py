from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ess_ope.envs.base import DiscreteFiniteHorizonEnv


@dataclass
class RandomMDPConfig:
    num_states: int = 100
    num_actions: int = 6
    horizon: int = 20
    branch_factor: int = 3
    linear_feature_dim: int = 12
    beta: float = 0.0
    reward_noise_std: float = 0.0
    seed: int = 0


class RandomMDP(DiscreteFiniteHorizonEnv):
    """Structured random finite-horizon MDP with sparse transitions."""

    def __init__(
        self,
        transition_probs: np.ndarray,
        rewards: np.ndarray,
        initial_state_dist: np.ndarray,
        horizon: int,
        linear_sa_features: np.ndarray,
        reward_linear_component: np.ndarray,
        reward_nonlinear_component: np.ndarray,
        beta: float,
        env_id: str,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(
            transition_probs=transition_probs,
            rewards=rewards,
            initial_state_dist=initial_state_dist,
            horizon=horizon,
            seed=seed,
        )
        self.linear_sa_features = linear_sa_features
        self.reward_linear_component = reward_linear_component
        self.reward_nonlinear_component = reward_nonlinear_component
        self.beta = float(beta)
        self.env_id = env_id

    @classmethod
    def generate(cls, config: RandomMDPConfig) -> "RandomMDP":
        rng = np.random.default_rng(config.seed)

        s = config.num_states
        a = config.num_actions
        h = config.horizon
        branch = min(max(1, config.branch_factor), s)

        transition_sa = np.zeros((s, a, s), dtype=float)
        for state in range(s):
            for action in range(a):
                support = rng.choice(s, size=branch, replace=False)
                probs = rng.dirichlet(np.ones(branch))
                transition_sa[state, action, support] = probs

        transition_probs = np.repeat(transition_sa[None, ...], h, axis=0)

        linear_features = rng.normal(size=(s, a, config.linear_feature_dim))
        w_linear = rng.normal(scale=1.0 / np.sqrt(config.linear_feature_dim), size=(config.linear_feature_dim,))
        linear_part = np.einsum("sad,d->sa", linear_features, w_linear)

        w_nl = rng.normal(scale=1.0 / np.sqrt(config.linear_feature_dim), size=(config.linear_feature_dim,))
        phase = rng.uniform(-np.pi, np.pi, size=(a,))
        nonlinear_part = np.sin(np.einsum("sad,d->sa", linear_features, w_nl) + phase[None, :])

        base_reward = linear_part + config.beta * nonlinear_part
        if config.reward_noise_std > 0:
            base_reward = base_reward + rng.normal(scale=config.reward_noise_std, size=base_reward.shape)
        base_reward = np.tanh(base_reward)

        rewards = np.repeat(base_reward[None, ...], h, axis=0)
        initial_state_dist = rng.dirichlet(np.ones(s))

        env_id = (
            f"randommdp_s{s}_a{a}_h{h}_b{branch}_d{config.linear_feature_dim}"
            f"_beta{config.beta:.2f}_seed{config.seed}"
        )

        return cls(
            transition_probs=transition_probs,
            rewards=rewards,
            initial_state_dist=initial_state_dist,
            horizon=h,
            linear_sa_features=linear_features,
            reward_linear_component=linear_part,
            reward_nonlinear_component=nonlinear_part,
            beta=config.beta,
            env_id=env_id,
            seed=config.seed,
        )
