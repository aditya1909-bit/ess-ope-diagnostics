from __future__ import annotations

import numpy as np

from ess_ope.data.generate import generate_offline_dataset
from ess_ope.evaluation.ground_truth import dynamic_programming_value
from ess_ope.policies.tabular import TabularPolicy
from ess_ope.v2.envs import StochasticTabularEnv, build_linear_features
from ess_ope.v2.estimators import evaluate_estimator
from ess_ope.v2.policies import build_policy_pair


def _tiny_env() -> StochasticTabularEnv:
    rewards = np.array([[[1.0, 0.0], [0.5, 2.0]]], dtype=float)
    reward_stds = np.zeros_like(rewards)
    transition = np.zeros((1, 2, 2, 2), dtype=float)
    transition[0, :, :, :] = np.eye(2)[None, :, :]
    return StochasticTabularEnv(
        transition_probs=transition,
        rewards=rewards,
        reward_stds=reward_stds,
        initial_state_dist=np.array([0.5, 0.5], dtype=float),
        horizon=1,
        linear_sa_features=build_linear_features(1, 2, 2, 6),
        env_id="tiny_env",
        seed=0,
    )


def test_v2_estimators_expose_contract_and_contributions() -> None:
    env = _tiny_env()
    target = TabularPolicy(np.array([[0.8, 0.2], [0.3, 0.7]], dtype=float))
    behavior = TabularPolicy(np.array([[0.6, 0.4], [0.5, 0.5]], dtype=float))
    truth = dynamic_programming_value(env, target)
    dataset = generate_offline_dataset(env, behavior, num_episodes=20, seed=7)

    for key in ["is", "wis", "pdis", "dr", "wdr", "fqe_linear", "dr_oracle"]:
        result = evaluate_estimator(
            estimator_key=key,
            dataset=dataset,
            target_policy=target,
            behavior_policy=behavior,
            env=env,
            gamma=1.0,
            truth_q=truth.q,
            truth_v=truth.v,
        )
        assert result.estimator_key == key
        assert isinstance(result.runtime_sec, float)
        if result.episode_contributions is not None:
            assert result.episode_contributions.shape == (dataset.num_episodes,)
            assert np.isclose(result.estimate, np.mean(result.episode_contributions))
        if key in {"is", "wis", "pdis"}:
            assert result.native_diagnostic_kind == "ess"
            assert result.native_diagnostic_value is not None
        else:
            assert result.native_diagnostic_kind is None


def test_v2_support_regimes_change_behavior_policy() -> None:
    env = _tiny_env()
    target = TabularPolicy(np.array([[0.8, 0.2], [0.3, 0.7]], dtype=float))
    truth = dynamic_programming_value(env, target)

    full = build_policy_pair(truth=truth, mismatch_level="medium", support_regime="full", seed=11)
    weak = build_policy_pair(truth=truth, mismatch_level="medium", support_regime="weak", seed=11)
    near = build_policy_pair(truth=truth, mismatch_level="medium", support_regime="near_violated", seed=11)

    assert not np.allclose(full.behavior_policy.probs, weak.behavior_policy.probs)
    assert not np.allclose(weak.behavior_policy.probs, near.behavior_policy.probs)
