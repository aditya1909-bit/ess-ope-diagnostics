from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Dict

import numpy as np

from ess_ope.data.dataset import EpisodeDataset
from ess_ope.metrics.confidence import IntervalEstimate, percentile_interval, resample_episodes, wald_mean_interval


@dataclass
class BootstrapSummary:
    samples: np.ndarray
    variance: float
    runtime_sec: float


def basic_interval(samples: np.ndarray, point_estimate: float, ci_level: float) -> IntervalEstimate:
    pct = percentile_interval(samples, ci_level=ci_level)
    low = 2.0 * float(point_estimate) - pct.high
    high = 2.0 * float(point_estimate) - pct.low
    center = float(point_estimate)
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
    boot = np.empty(int(n_boot), dtype=float)
    start = perf_counter()
    for i in range(int(n_boot)):
        idx = rng.integers(0, k, size=draw_size)
        sample_ds = resample_episodes(dataset, idx)
        boot[i] = float(estimate_fn(sample_ds))
    runtime = perf_counter() - start
    variance = float(np.var(boot, ddof=1)) if boot.size > 1 else 0.0
    return BootstrapSummary(samples=boot, variance=variance, runtime_sec=runtime)


def summarize_intervals(
    point_estimate: float,
    contributions: np.ndarray | None,
    bootstrap: BootstrapSummary | None,
    ci_level: float,
) -> Dict[str, IntervalEstimate]:
    out: Dict[str, IntervalEstimate] = {}
    if contributions is not None and np.asarray(contributions).size > 0:
        out["analytic"] = wald_mean_interval(np.asarray(contributions, dtype=float), ci_level=ci_level)
    if bootstrap is not None and bootstrap.samples.size > 0:
        out["bootstrap_percentile"] = percentile_interval(bootstrap.samples, ci_level=ci_level)
        out["bootstrap_basic"] = basic_interval(bootstrap.samples, point_estimate=point_estimate, ci_level=ci_level)
    return out
