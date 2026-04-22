from __future__ import annotations

import numpy as np

from ess_ope.evaluation.ground_truth import dynamic_programming_value
from ess_ope.study.envs import BEHAVIOR_FLOOR, build_contextual_bandit, build_long_horizon_mdp, build_short_horizon_mdp


def test_study_environment_builders_match_paper_shapes() -> None:
    bandit = build_contextual_bandit(seed=7, mismatch_level="medium", reward_variance_regime="high", reward_mean_structure="smooth")
    assert bandit.env.horizon == 1
    assert bandit.env.num_states == 20
    assert bandit.env.num_actions == 4
    assert np.all(bandit.behavior_policy.probs >= BEHAVIOR_FLOOR - 1e-8)

    short_a = build_short_horizon_mdp(seed=4, mismatch_level="medium", reward_variant="dense")
    short_b = build_short_horizon_mdp(seed=4, mismatch_level="medium", reward_variant="dense")
    assert short_a.env.horizon == 5
    assert short_a.env.num_states == 30
    assert short_a.env.num_actions == 4
    truth_a = dynamic_programming_value(short_a.env, short_a.target_policy)
    truth_b = dynamic_programming_value(short_b.env, short_b.target_policy)
    assert np.isclose(truth_a.value, truth_b.value)

    long_env = build_long_horizon_mdp(seed=11, mismatch_level="high")
    assert long_env.env.horizon == 20
    assert long_env.env.num_states == 50
    assert long_env.env.num_actions == 4
    assert dynamic_programming_value(long_env.env, long_env.target_policy).value == dynamic_programming_value(long_env.env, long_env.target_policy).value


def test_bandit_reward_structures_hold_target_value_fixed() -> None:
    smooth = build_contextual_bandit(seed=19, mismatch_level="medium", reward_variance_regime="medium", reward_mean_structure="smooth")
    hetero = build_contextual_bandit(seed=19, mismatch_level="medium", reward_variance_regime="medium", reward_mean_structure="heterogeneous")
    smooth_truth = dynamic_programming_value(smooth.env, smooth.target_policy)
    hetero_truth = dynamic_programming_value(hetero.env, hetero.target_policy)
    assert np.isclose(smooth_truth.value, hetero_truth.value, atol=1e-8)
    assert not np.allclose(smooth.env.rewards, hetero.env.rewards)


def test_fqe_calibration_variants_have_intended_roles() -> None:
    from ess_ope.study.envs import build_environment_bundle

    short = build_environment_bundle({"name": "fqe_calibration_pair"}, seed=3, condition={"calibration_env": "short_positive"})
    long = build_environment_bundle({"name": "fqe_calibration_pair"}, seed=3, condition={"calibration_env": "long_stress"})
    assert short.env.horizon == 5
    assert short.metadata["calibration_role"] == "positive"
    assert long.env.horizon == 12
    assert long.metadata["calibration_role"] == "stress"
