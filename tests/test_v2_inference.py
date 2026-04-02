from __future__ import annotations

import itertools

import numpy as np

from ess_ope.data.dataset import EpisodeDataset
from ess_ope.v2.inference import episode_bootstrap


def test_v2_episode_bootstrap_resamples_whole_episodes() -> None:
    dataset = EpisodeDataset(
        states=np.array([[0, 0], [1, 1]], dtype=int),
        actions=np.array([[0, 1], [1, 0]], dtype=int),
        rewards=np.array([[1.0, 2.0], [10.0, 20.0]], dtype=float),
        next_states=np.array([[0, 0], [1, 1]], dtype=int),
        dones=np.array([[False, True], [False, True]], dtype=bool),
    )

    def estimate_fn(ds: EpisodeDataset) -> float:
        episode_returns = ds.rewards.sum(axis=1)
        return float(np.mean(episode_returns))

    summary = episode_bootstrap(dataset, estimate_fn=estimate_fn, n_boot=10, seed=3)
    valid_values = set()
    returns = [3.0, 30.0]
    for idx in itertools.product([0, 1], repeat=2):
        valid_values.add(float(np.mean([returns[i] for i in idx])))

    assert summary.samples.size == 10
    assert set(np.round(summary.samples, 8)).issubset(set(np.round(list(valid_values), 8)))
