from __future__ import annotations

import numpy as np

from ess_ope.data.dataset import EpisodeDataset
from ess_ope.study.inference import bootstrap_normal_interval, episode_bootstrap, summarize_intervals


def _episode_marker_dataset() -> EpisodeDataset:
    return EpisodeDataset(
        states=np.array([[0, 0], [1, 1], [2, 2]], dtype=int),
        actions=np.array([[0, 0], [1, 1], [2, 2]], dtype=int),
        rewards=np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=float),
        next_states=np.array([[0, 0], [1, 1], [2, 2]], dtype=int),
        dones=np.array([[False, True], [False, True], [False, True]], dtype=bool),
    )


def test_episode_bootstrap_preserves_episode_structure() -> None:
    dataset = _episode_marker_dataset()
    bootstrap = episode_bootstrap(
        dataset=dataset,
        estimate_fn=lambda ds: float(np.mean(ds.rewards[:, 1] - 10.0 * ds.rewards[:, 0])),
        n_boot=8,
        seed=0,
        subsample_ratio=1.0,
    )
    assert bootstrap.samples.shape == (8,)
    assert np.allclose(bootstrap.samples, 0.0)
    normal = bootstrap_normal_interval(bootstrap.samples, point_estimate=0.0, ci_level=0.9)
    assert np.isclose(normal.center, 0.0)
    intervals = summarize_intervals(
        point_estimate=0.0,
        bootstrap=bootstrap,
        contributions=np.array([0.0, 0.0, 0.0]),
        ci_level=0.9,
        methods=["bootstrap_percentile", "bootstrap_normal", "concentration_empirical_bernstein"],
    )
    assert set(intervals) == {"bootstrap_percentile", "bootstrap_normal", "concentration_empirical_bernstein"}
