from __future__ import annotations

import numpy as np
import pytest

from ess_ope.envs.chain_bandit import ChainBanditConfig, ChainBanditEnv
from ess_ope.evaluation.ground_truth import dynamic_programming_value, monte_carlo_policy_value
from ess_ope.policies.tabular import TabularPolicy


def _uniform_time_varying_policy(env: ChainBanditEnv) -> TabularPolicy:
    probs = np.full((env.horizon, env.num_states, env.num_actions), 1.0 / env.num_actions, dtype=float)
    return TabularPolicy(probs)


def test_chain_bandit_shapes_and_normalization() -> None:
    env = ChainBanditEnv.generate(
        ChainBanditConfig(
            num_states=4,
            num_actions=3,
            horizon=6,
            linear_feature_dim=9,
            transition_strength=0.7,
            reward_mean_scale=1.5,
            reward_gap=0.5,
            reward_std=0.25,
            beta=0.2,
            variant="transitional",
            seed=7,
        )
    )

    assert env.transition_probs.shape == (6, 4, 3, 4)
    assert env.rewards.shape == (6, 4, 3)
    assert env.linear_sa_features.shape == (6, 4, 3, 9)
    assert np.allclose(env.transition_probs.sum(axis=-1), 1.0, atol=1e-6)
    assert np.isclose(env.initial_state_dist.sum(), 1.0, atol=1e-6)


def test_reward_only_variant_has_action_independent_transitions() -> None:
    env = ChainBanditEnv.generate(
        ChainBanditConfig(
            num_states=3,
            num_actions=4,
            horizon=5,
            transition_strength=1.0,
            reward_mean_scale=1.0,
            reward_gap=0.5,
            reward_std=0.2,
            beta=0.4,
            variant="reward_only",
            seed=11,
        )
    )

    first_action = env.transition_probs[:, :, :1, :]
    assert np.allclose(env.transition_probs, first_action, atol=1e-6)


def test_reward_std_does_not_change_expected_value() -> None:
    base_cfg = dict(
        num_states=3,
        num_actions=3,
        horizon=4,
        linear_feature_dim=8,
        transition_strength=0.8,
        reward_mean_scale=1.2,
        reward_gap=0.7,
        beta=0.3,
        variant="transitional",
        seed=19,
    )
    env_low = ChainBanditEnv.generate(ChainBanditConfig(**base_cfg, reward_std=0.1))
    env_high = ChainBanditEnv.generate(ChainBanditConfig(**base_cfg, reward_std=2.0))
    policy = _uniform_time_varying_policy(env_low)

    value_low = dynamic_programming_value(env_low, policy).value
    value_high = dynamic_programming_value(env_high, policy).value
    assert np.isclose(value_low, value_high, atol=1e-9)


def test_dynamic_programming_matches_monte_carlo() -> None:
    env = ChainBanditEnv.generate(
        ChainBanditConfig(
            num_states=3,
            num_actions=2,
            horizon=5,
            linear_feature_dim=8,
            transition_strength=0.6,
            reward_mean_scale=0.8,
            reward_gap=0.4,
            reward_std=0.1,
            beta=0.1,
            variant="transitional",
            seed=5,
        )
    )
    policy = _uniform_time_varying_policy(env)

    dp_value = dynamic_programming_value(env, policy).value
    mc_value = monte_carlo_policy_value(env, policy, num_episodes=5000, seed=123).value
    assert abs(dp_value - mc_value) < 0.12


@pytest.mark.parametrize("feature_dim", [1, 8, 9, 10, 12])
def test_chain_bandit_feature_dim_matches_requested_width(feature_dim: int) -> None:
    env = ChainBanditEnv.generate(
        ChainBanditConfig(
            num_states=3,
            num_actions=2,
            horizon=4,
            linear_feature_dim=feature_dim,
            transition_strength=0.5,
            reward_mean_scale=1.0,
            reward_gap=0.4,
            reward_std=0.1,
            beta=0.2,
            variant="transitional",
            seed=13,
        )
    )

    assert env.linear_sa_features.shape == (4, 3, 2, feature_dim)


def test_chain_bandit_feature_dim_must_be_positive() -> None:
    with pytest.raises(ValueError, match="linear_feature_dim must be positive"):
        ChainBanditEnv.generate(
            ChainBanditConfig(
                num_states=3,
                num_actions=2,
                horizon=4,
                linear_feature_dim=0,
                transition_strength=0.5,
                reward_mean_scale=1.0,
                reward_gap=0.4,
                reward_std=0.1,
                beta=0.2,
                variant="transitional",
                seed=13,
            )
        )
