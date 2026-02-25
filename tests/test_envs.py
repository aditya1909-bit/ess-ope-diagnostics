import numpy as np

from ess_ope.data.generate import generate_offline_dataset
from ess_ope.envs.random_mdp import RandomMDP, RandomMDPConfig
from ess_ope.policies.tabular import TabularPolicy


def test_random_mdp_transition_rows_sum_to_one() -> None:
    env = RandomMDP.generate(RandomMDPConfig(num_states=40, num_actions=5, horizon=12, seed=3))
    row_sums = env.transition_probs.sum(axis=-1)
    assert np.allclose(row_sums, 1.0, atol=1e-6)


def test_random_mdp_rewards_in_range() -> None:
    env = RandomMDP.generate(RandomMDPConfig(num_states=30, num_actions=4, horizon=10, beta=1.0, seed=2))
    assert np.max(env.rewards) <= 1.0 + 1e-8
    assert np.min(env.rewards) >= -1.0 - 1e-8


def test_rollout_dataset_shapes() -> None:
    env = RandomMDP.generate(RandomMDPConfig(num_states=25, num_actions=4, horizon=8, seed=5))
    behavior = TabularPolicy.uniform(env.num_states, env.num_actions)
    data = generate_offline_dataset(env, behavior, num_episodes=17, seed=11)

    assert data.states.shape == (17, env.horizon)
    assert data.actions.shape == (17, env.horizon)
    assert data.rewards.shape == (17, env.horizon)
    assert data.next_states.shape == (17, env.horizon)
    assert data.dones.shape == (17, env.horizon)
