from __future__ import annotations

import numpy as np

from ess_ope.data.dataset import EpisodeDataset
from ess_ope.estimators.dr import doubly_robust_episode_contributions, doubly_robust_estimate
from ess_ope.metrics.confidence import bootstrap_estimator_interval, wald_mean_interval
from ess_ope.policies.tabular import TabularPolicy


def test_oracle_dr_contributions_average_to_the_estimate() -> None:
    dataset = EpisodeDataset(
        states=np.array([[0], [1], [0], [1]], dtype=int),
        actions=np.array([[0], [1], [1], [0]], dtype=int),
        rewards=np.array([[1.0], [0.0], [2.0], [3.0]], dtype=float),
        next_states=np.array([[0], [0], [0], [0]], dtype=int),
        dones=np.array([[True], [True], [True], [True]], dtype=bool),
    )
    target = TabularPolicy(np.array([[0.7, 0.3], [0.2, 0.8]], dtype=float))
    behavior = TabularPolicy(np.array([[0.5, 0.5], [0.5, 0.5]], dtype=float))
    q_hat = np.array([[[1.0, 2.0], [3.0, 0.0]]], dtype=float)
    v_hat = np.array([[1.3, 0.6], [0.0, 0.0]], dtype=float)

    contrib = doubly_robust_episode_contributions(
        dataset=dataset,
        target_policy=target,
        behavior_policy=behavior,
        q_hat=q_hat,
        v_hat=v_hat,
    )
    estimate = doubly_robust_estimate(
        dataset=dataset,
        target_policy=target,
        behavior_policy=behavior,
        q_hat=q_hat,
        v_hat=v_hat,
    )

    assert contrib.shape == (dataset.num_episodes,)
    assert np.isclose(estimate, np.mean(contrib))


def test_confidence_interval_helpers_return_finite_bounds() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    analytic = wald_mean_interval(values, ci_level=0.95)
    assert analytic.low < analytic.center < analytic.high
    assert analytic.width > 0.0

    dataset = EpisodeDataset(
        states=np.array([[0], [0], [0], [0]], dtype=int),
        actions=np.array([[0], [0], [0], [0]], dtype=int),
        rewards=np.array([[1.0], [2.0], [3.0], [4.0]], dtype=float),
        next_states=np.array([[0], [0], [0], [0]], dtype=int),
        dones=np.array([[True], [True], [True], [True]], dtype=bool),
    )
    interval = bootstrap_estimator_interval(
        dataset=dataset,
        estimate_fn=lambda ds: float(np.mean(ds.rewards)),
        n_boot=20,
        ci_level=0.95,
        seed=7,
    )
    assert np.isfinite(interval.low)
    assert np.isfinite(interval.high)
    assert interval.low <= interval.high
