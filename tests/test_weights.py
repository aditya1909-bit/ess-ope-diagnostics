import numpy as np

from ess_ope.data.generate import generate_offline_dataset
from ess_ope.estimators import compute_importance_weights
from ess_ope.envs.random_mdp import RandomMDP, RandomMDPConfig
from ess_ope.metrics.ess import episode_ess
from ess_ope.policies.tabular import TabularPolicy


def test_weights_all_one_when_policies_equal() -> None:
    env = RandomMDP.generate(RandomMDPConfig(num_states=20, num_actions=4, horizon=6, seed=0))
    policy = TabularPolicy.uniform(env.num_states, env.num_actions)
    data = generate_offline_dataset(env, policy, num_episodes=50, seed=42)

    weights = compute_importance_weights(data, policy, policy)
    assert np.allclose(weights.episode_weights, 1.0)
    assert np.isclose(episode_ess(weights.episode_weights), data.num_episodes)


def test_weights_spike_when_target_more_peaked_than_behavior() -> None:
    env = RandomMDP.generate(RandomMDPConfig(num_states=15, num_actions=4, horizon=5, seed=7))
    behavior = TabularPolicy.uniform(env.num_states, env.num_actions)

    target_probs = np.full((env.num_states, env.num_actions), 0.01)
    target_probs[:, 0] = 0.97
    target_probs = target_probs / target_probs.sum(axis=-1, keepdims=True)
    target = TabularPolicy(target_probs)

    data = generate_offline_dataset(env, behavior, num_episodes=200, seed=13)
    weights = compute_importance_weights(data, target, behavior)

    assert np.std(weights.episode_weights) > 0.0
    assert episode_ess(weights.episode_weights) < data.num_episodes
