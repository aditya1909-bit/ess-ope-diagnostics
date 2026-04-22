from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Dict, Iterable

import numpy as np
from scipy import stats

from ess_ope.data.dataset import EpisodeDataset
from ess_ope.metrics.confidence import IntervalEstimate, percentile_interval, resample_episodes


@dataclass
class BootstrapSummary:
    samples: np.ndarray
    variance: float
    runtime_sec: float


def bootstrap_normal_interval(samples: np.ndarray, point_estimate: float, ci_level: float) -> IntervalEstimate:
    arr = np.asarray(samples, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return IntervalEstimate(low=np.nan, high=np.nan, width=np.nan, center=np.nan)
    alpha = 1.0 - float(ci_level)
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    se = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    low = float(point_estimate) - z * se
    high = float(point_estimate) + z * se
    return IntervalEstimate(low=low, high=high, width=high - low, center=float(point_estimate))


def empirical_bernstein_interval(values: np.ndarray, ci_level: float) -> IntervalEstimate:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return IntervalEstimate(low=np.nan, high=np.nan, width=np.nan, center=np.nan)
    center = float(np.mean(arr))
    if arr.size == 1:
        return IntervalEstimate(low=center, high=center, width=0.0, center=center)
    alpha = max(1e-12, 1.0 - float(ci_level))
    variance = float(np.var(arr, ddof=1))
    radius = float(np.max(np.abs(arr - center)))
    bonus = np.sqrt(2.0 * variance * np.log(3.0 / alpha) / arr.size)
    bonus += 3.0 * radius * np.log(3.0 / alpha) / max(1.0, arr.size - 1.0)
    low = center - bonus
    high = center + bonus
    return IntervalEstimate(low=low, high=high, width=high - low, center=center)


def episode_bootstrap(
    dataset: EpisodeDataset,
    estimate_fn: Callable[[EpisodeDataset], float],
    n_boot: int,
    seed: int,
    subsample_ratio: float = 1.0,
) -> BootstrapSummary:
    if int(n_boot) <= 0:
        return BootstrapSummary(samples=np.array([], dtype=float), variance=np.nan, runtime_sec=0.0)
    rng = np.random.default_rng(seed)
    k = dataset.num_episodes
    draw_size = max(1, int(round(float(subsample_ratio) * k)))
    samples = np.empty(int(n_boot), dtype=float)
    start = perf_counter()
    for idx in range(int(n_boot)):
        boot_idx = rng.integers(0, k, size=draw_size)
        samples[idx] = float(estimate_fn(resample_episodes(dataset, boot_idx)))
    runtime_sec = perf_counter() - start
    variance = float(np.var(samples, ddof=1)) if samples.size > 1 else 0.0
    return BootstrapSummary(samples=samples, variance=variance, runtime_sec=runtime_sec)


def summarize_intervals(
    point_estimate: float,
    bootstrap: BootstrapSummary | None,
    contributions: np.ndarray | None,
    ci_level: float,
    methods: Iterable[str],
) -> Dict[str, IntervalEstimate]:
    out: Dict[str, IntervalEstimate] = {}
    method_set = set(methods)
    if bootstrap is not None and bootstrap.samples.size > 0:
        if "bootstrap_percentile" in method_set:
            out["bootstrap_percentile"] = percentile_interval(bootstrap.samples, ci_level=ci_level)
        if "bootstrap_normal" in method_set:
            out["bootstrap_normal"] = bootstrap_normal_interval(bootstrap.samples, point_estimate=point_estimate, ci_level=ci_level)
    if contributions is not None and contributions.size > 0 and "concentration_empirical_bernstein" in method_set:
        out["concentration_empirical_bernstein"] = empirical_bernstein_interval(contributions, ci_level=ci_level)
    return out
