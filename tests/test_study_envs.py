from __future__ import annotations

import numpy as np

from ess_ope.evaluation.ground_truth import dynamic_programming_value
from ess_ope.study.envs import BEHAVIOR_FLOOR, build_contextual_bandit, build_fqe_case_study, build_long_horizon_mdp, build_short_horizon_mdp


def test_study_environment_builders_match_paper_shapes() -> None:
    bandit = build_contextual_bandit(seed=7, mismatch_alpha=0.4, reward_variance_scale=4.0)
    assert bandit.env.horizon == 1
    assert bandit.env.num_states == 10
    assert bandit.env.num_actions == 5
    assert np.all(bandit.behavior_policy.probs >= BEHAVIOR_FLOOR - 1e-8)

    short_a = build_short_horizon_mdp(seed=4, mismatch_alpha=0.4)
    short_b = build_short_horizon_mdp(seed=4, mismatch_alpha=0.4)
    assert short_a.env.horizon == 5
    assert short_a.env.num_states == 30
    assert short_a.env.num_actions == 3
    truth_a = dynamic_programming_value(short_a.env, short_a.target_policy)
    truth_b = dynamic_programming_value(short_b.env, short_b.target_policy)
    assert np.isclose(truth_a.value, truth_b.value)

    long_env = build_long_horizon_mdp(seed=11, mismatch_alpha=0.8)
    assert long_env.env.horizon == 20
    assert long_env.env.num_states == 40
    assert long_env.env.num_actions == 3
    assert dynamic_programming_value(long_env.env, long_env.target_policy).value == dynamic_programming_value(long_env.env, long_env.target_policy).value


def test_bandit_reward_variance_intervention_holds_weights_and_means_fixed() -> None:
    low = build_contextual_bandit(seed=19, mismatch_alpha=0.4, reward_variance_scale=0.1)
    high = build_contextual_bandit(seed=19, mismatch_alpha=0.4, reward_variance_scale=4.0)
    low_truth = dynamic_programming_value(low.env, low.target_policy)
    high_truth = dynamic_programming_value(high.env, high.target_policy)
    assert np.isclose(low_truth.value, high_truth.value, atol=1e-8)
    assert np.allclose(low.env.rewards, high.env.rewards)
    assert not np.allclose(low.env.reward_stds, high.env.reward_stds)
    assert np.allclose(low.behavior_policy.probs, high.behavior_policy.probs)
    assert np.allclose(low.target_policy.probs, high.target_policy.probs)


def test_fqe_calibration_variants_have_intended_roles() -> None:
    short = build_fqe_case_study(seed=3, calibration_env="short_horizon")
    long = build_fqe_case_study(seed=3, calibration_env="long_horizon")
    assert short.env.horizon == 5
    assert short.metadata["calibration_env"] == "short_horizon"
    assert long.env.horizon == 20
    assert long.metadata["calibration_env"] == "long_horizon"
