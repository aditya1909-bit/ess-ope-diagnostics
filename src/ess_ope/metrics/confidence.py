from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import stats

from ess_ope.data.dataset import EpisodeDataset


@dataclass
class IntervalEstimate:
    low: float
    high: float
    width: float
    center: float


def resample_episodes(dataset: EpisodeDataset, indices: np.ndarray) -> EpisodeDataset:
    idx = np.asarray(indices, dtype=int)
    return EpisodeDataset(
        states=dataset.states[idx],
        actions=dataset.actions[idx],
        rewards=dataset.rewards[idx],
        next_states=dataset.next_states[idx],
        dones=dataset.dones[idx],
    )


def wald_mean_interval(values: np.ndarray, ci_level: float = 0.95) -> IntervalEstimate:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return IntervalEstimate(low=np.nan, high=np.nan, width=np.nan, center=np.nan)

    center = float(np.mean(arr))
    if arr.size == 1:
        return IntervalEstimate(low=center, high=center, width=0.0, center=center)

    alpha = 1.0 - float(ci_level)
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    stderr = float(np.std(arr, ddof=1) / np.sqrt(arr.size))
    low = center - z * stderr
    high = center + z * stderr
    return IntervalEstimate(low=low, high=high, width=high - low, center=center)


def percentile_interval(samples: np.ndarray, ci_level: float = 0.95) -> IntervalEstimate:
    arr = np.asarray(samples, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return IntervalEstimate(low=np.nan, high=np.nan, width=np.nan, center=np.nan)

    alpha = 1.0 - float(ci_level)
    low = float(np.quantile(arr, alpha / 2.0))
    high = float(np.quantile(arr, 1.0 - alpha / 2.0))
    center = float(np.mean(arr))
    return IntervalEstimate(low=low, high=high, width=high - low, center=center)


def bootstrap_estimator_interval(
    dataset: EpisodeDataset,
    estimate_fn: Callable[[EpisodeDataset], float],
    n_boot: int,
    ci_level: float = 0.95,
    seed: int = 0,
) -> IntervalEstimate:
    if int(n_boot) <= 0:
        return IntervalEstimate(low=np.nan, high=np.nan, width=np.nan, center=np.nan)

    rng = np.random.default_rng(seed)
    k = dataset.num_episodes
    boot = np.empty(int(n_boot), dtype=float)
    for i in range(int(n_boot)):
        idx = rng.integers(0, k, size=k)
        boot[i] = float(estimate_fn(resample_episodes(dataset, idx)))
    return percentile_interval(boot, ci_level=ci_level)
