from __future__ import annotations

import numpy as np

from ess_ope.data.dataset import EpisodeDataset
from ess_ope.estimators.dm_fqe import direct_model_tabular
from ess_ope.policies.tabular import TabularPolicy
from ess_ope.study.envs import StochasticTabularEnv, build_linear_features
from ess_ope.study.estimators import evaluate_estimator


def _tiny_env() -> StochasticTabularEnv:
    transition = np.zeros((1, 2, 2, 2), dtype=float)
    transition[0, 0, :, 0] = 1.0
    transition[0, 1, :, 1] = 1.0
    rewards = np.array([[[1.0, 0.0], [0.0, 2.0]]], dtype=float)
    reward_stds = np.zeros_like(rewards)
    return StochasticTabularEnv(
        transition_probs=transition,
        rewards=rewards,
        reward_stds=reward_stds,
        initial_state_dist=np.array([0.5, 0.5], dtype=float),
        horizon=1,
        linear_sa_features=build_linear_features(1, 2, 2, 6),
        env_id="tiny",
        seed=0,
    )


def _fully_observed_bandit_dataset() -> EpisodeDataset:
    return EpisodeDataset(
        states=np.array([[0], [0], [1], [1]], dtype=int),
        actions=np.array([[0], [1], [0], [1]], dtype=int),
        rewards=np.array([[1.0], [0.0], [0.0], [2.0]], dtype=float),
        next_states=np.array([[0], [0], [1], [1]], dtype=int),
        dones=np.array([[True], [True], [True], [True]], dtype=bool),
    )


def test_paper_estimators_behave_on_tiny_fully_observed_fixture() -> None:
    env = _tiny_env()
    dataset = _fully_observed_bandit_dataset()
    behavior_uniform = TabularPolicy(np.full((2, 2), 0.5, dtype=float))
    target_greedy = TabularPolicy(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float))
    target_uniform = TabularPolicy(np.full((2, 2), 0.5, dtype=float))

    is_res = evaluate_estimator("is", dataset, target_greedy, behavior_uniform, env)
    snis_res = evaluate_estimator("snis", dataset, target_greedy, behavior_uniform, env)
    pdis_res = evaluate_estimator("pdis", dataset, target_greedy, behavior_uniform, env)
    assert np.isclose(is_res.estimate, 1.5)
    assert np.isclose(snis_res.estimate, 1.5)
    assert np.isclose(pdis_res.estimate, 1.5)
    assert is_res.wess_native_applicable
    assert np.isclose(is_res.shared_wess, 2.0)

    dm_truth = direct_model_tabular(dataset, target_greedy, num_states=2, num_actions=2, horizon=1, initial_state_dist=env.initial_state_dist)
    dm_res = evaluate_estimator("dm", dataset, target_greedy, behavior_uniform, env)
    fqe_res = evaluate_estimator("fqe", dataset, target_greedy, behavior_uniform, env)
    assert np.isclose(dm_res.estimate, dm_truth.value)
    assert np.isclose(dm_res.estimate, 1.5)
    assert np.isclose(fqe_res.estimate, 1.5, atol=1e-4)

    dr_res = evaluate_estimator("dr", dataset, target_uniform, behavior_uniform, env)
    dm_uniform = evaluate_estimator("dm", dataset, target_uniform, behavior_uniform, env)
    assert np.isclose(dr_res.estimate, dm_uniform.estimate, atol=1e-6)

    mrdr_res = evaluate_estimator("mrdr", dataset, target_greedy, behavior_uniform, env)
    assert np.isfinite(mrdr_res.estimate)
    assert not mrdr_res.wess_native_applicable
