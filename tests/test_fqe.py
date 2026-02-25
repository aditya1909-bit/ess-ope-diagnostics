import numpy as np

from ess_ope.data.dataset import EpisodeDataset
from ess_ope.estimators.dm_fqe import fitted_q_evaluation
from ess_ope.envs.base import DiscreteFiniteHorizonEnv
from ess_ope.evaluation.ground_truth import dynamic_programming_value
from ess_ope.policies.tabular import TabularPolicy


def test_tabular_fqe_matches_dp_in_tiny_fully_covered_case() -> None:
    horizon, num_states, num_actions = 1, 2, 2

    transition = np.zeros((horizon, num_states, num_actions, num_states), dtype=float)
    transition[0, :, :, 0] = 1.0

    rewards = np.array([[[0.1, 0.2], [0.3, 0.4]]], dtype=float)
    init = np.array([0.5, 0.5], dtype=float)

    env = DiscreteFiniteHorizonEnv(
        transition_probs=transition,
        rewards=rewards,
        initial_state_dist=init,
        horizon=horizon,
    )

    policy = TabularPolicy(np.array([[0.25, 0.75], [0.6, 0.4]], dtype=float))
    truth = dynamic_programming_value(env, policy)

    states = []
    actions = []
    rewards_list = []
    next_states = []
    dones = []

    repeats_per_pair = 30
    for s in range(num_states):
        for a in range(num_actions):
            for _ in range(repeats_per_pair):
                states.append([s])
                actions.append([a])
                rewards_list.append([rewards[0, s, a]])
                next_states.append([0])
                dones.append([True])

    data = EpisodeDataset(
        states=np.array(states, dtype=int),
        actions=np.array(actions, dtype=int),
        rewards=np.array(rewards_list, dtype=float),
        next_states=np.array(next_states, dtype=int),
        dones=np.array(dones, dtype=bool),
    )

    fqe = fitted_q_evaluation(
        dataset=data,
        target_policy=policy,
        num_states=num_states,
        num_actions=num_actions,
        horizon=horizon,
        initial_state_dist=init,
        model_type="tabular",
        l2_reg=0.0,
    )

    assert np.isclose(fqe.value, truth.value, atol=1e-8)
    assert np.allclose(fqe.q, truth.q, atol=1e-8)
