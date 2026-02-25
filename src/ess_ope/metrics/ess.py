from __future__ import annotations

from typing import Dict

import numpy as np


def normalized_weights(weights: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    total = np.sum(w)
    if total <= eps:
        return np.full_like(w, 1.0 / len(w))
    return w / total


def episode_ess(weights: np.ndarray, eps: float = 1e-12) -> float:
    w = np.asarray(weights, dtype=float)
    denom = np.sum(w**2)
    numer = np.sum(w) ** 2
    if denom <= eps:
        return 0.0
    return float(numer / denom)


def weight_cv(weights: np.ndarray, eps: float = 1e-12) -> float:
    w = np.asarray(weights, dtype=float)
    mean = np.mean(w)
    if abs(mean) <= eps:
        return 0.0
    return float(np.std(w) / mean)


def weight_entropy(weights: np.ndarray, eps: float = 1e-12) -> float:
    p = normalized_weights(weights, eps=eps)
    p = np.clip(p, eps, 1.0)
    return float(-np.sum(p * np.log(p)))


def weight_perplexity(weights: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.exp(weight_entropy(weights, eps=eps)))


def weight_summary(weights: np.ndarray) -> Dict[str, float]:
    p = normalized_weights(weights)
    return {
        "ess_is": episode_ess(weights),
        "max_weight_share": float(np.max(p)),
        "weight_cv": weight_cv(weights),
        "weight_entropy": weight_entropy(weights),
        "weight_perplexity": weight_perplexity(weights),
    }
