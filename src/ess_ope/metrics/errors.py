from __future__ import annotations

from typing import Dict

import numpy as np


def absolute_error(estimate: float, truth: float) -> float:
    return float(abs(estimate - truth))


def squared_error(estimate: float, truth: float) -> float:
    return float((estimate - truth) ** 2)


def point_error_metrics(estimate: float, truth: float) -> Dict[str, float]:
    err = float(estimate - truth)
    return {
        "error": err,
        "abs_error": abs(err),
        "squared_error": err**2,
    }


def aggregate_error_metrics(estimates: np.ndarray, truth: float) -> Dict[str, float]:
    est = np.asarray(estimates, dtype=float)
    err = est - truth
    return {
        "bias": float(np.mean(err)),
        "mse": float(np.mean(err**2)),
        "variance": float(np.var(est)),
        "mae": float(np.mean(np.abs(err))),
    }
