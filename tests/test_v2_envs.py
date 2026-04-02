from __future__ import annotations

import numpy as np

from ess_ope.evaluation.ground_truth import dynamic_programming_value
from ess_ope.policies.tabular import TabularPolicy
from ess_ope.v2.envs import StochasticTabularEnv, build_linear_features


def test_v2_bandit_truth_matches_closed_form_expectation() -> None:
    rewards = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=float)
    reward_stds = np.zeros_like(rewards)
    transition = np.zeros((1, 2, 2, 2), dtype=float)
    transition[0, :, :, :] = np.eye(2)[None, :, :]
    init = np.array([0.25, 0.75], dtype=float)
    env = StochasticTabularEnv(
        transition_probs=transition,
        rewards=rewards,
        reward_stds=reward_stds,
        initial_state_dist=init,
        horizon=1,
        linear_sa_features=build_linear_features(1, 2, 2, 6),
        env_id="tiny_bandit",
        seed=0,
    )
    policy = TabularPolicy(np.array([[0.2, 0.8], [0.7, 0.3]], dtype=float))
    truth = dynamic_programming_value(env, policy)

    expected = 0.25 * (0.2 * 1.0 + 0.8 * 2.0) + 0.75 * (0.7 * 3.0 + 0.3 * 4.0)
    assert np.isclose(truth.value, expected)
