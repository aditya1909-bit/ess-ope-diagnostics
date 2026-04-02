from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from ess_ope.evaluation.ground_truth import GroundTruthResult
from ess_ope.policies.tabular import TabularPolicy


MISMATCH_MIX = {
    "low": 0.15,
    "medium": 0.35,
    "high": 0.6,
    "extreme": 0.82,
}

SUPPORT_FLOOR = {
    "full": 0.08,
    "weak": 0.03,
    "near_violated": 0.005,
}

TARGET_ACTION_STRESS = {
    "full": 1.0,
    "weak": 0.35,
    "near_violated": 0.08,
}


@dataclass
class PolicyBundle:
    target_policy: TabularPolicy
    behavior_policy: TabularPolicy
    metadata: Dict[str, float]


def _softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = np.asarray(logits, dtype=float) / max(temperature, 1e-8)
    shifted = scaled - np.max(scaled, axis=-1, keepdims=True)
    probs = np.exp(shifted)
    probs /= np.sum(probs, axis=-1, keepdims=True)
    return probs


def _apply_support_floor(probs: np.ndarray, floor: float) -> np.ndarray:
    clipped = np.maximum(np.asarray(probs, dtype=float), float(floor))
    clipped /= np.sum(clipped, axis=-1, keepdims=True)
    return clipped


def _anti_policy_probs(target: np.ndarray) -> np.ndarray:
    rev = np.max(target, axis=-1, keepdims=True) - target + 1e-3
    rev /= np.sum(rev, axis=-1, keepdims=True)
    return rev


def _apply_support_stress(behavior: np.ndarray, target: np.ndarray, support_regime: str) -> np.ndarray:
    stressed = np.asarray(behavior, dtype=float).copy()
    stress = float(TARGET_ACTION_STRESS.get(str(support_regime), 0.35))
    greedy = np.argmax(target, axis=-1)

    if stressed.ndim == 2:
        stressed[np.arange(stressed.shape[0]), greedy] *= stress
    else:
        t_idx, s_idx = np.indices(greedy.shape)
        stressed[t_idx, s_idx, greedy] *= stress

    stressed /= np.sum(stressed, axis=-1, keepdims=True)
    return stressed


def build_policy_pair(
    truth: GroundTruthResult,
    mismatch_level: str,
    support_regime: str,
    seed: int,
    target_temperature: float = 0.8,
) -> PolicyBundle:
    rng = np.random.default_rng(seed)
    q = np.asarray(truth.q, dtype=float)
    target_probs = _softmax(q, temperature=target_temperature)
    target_probs = _apply_support_floor(target_probs, 1e-4)

    anti = _anti_policy_probs(target_probs)
    mix = float(MISMATCH_MIX.get(str(mismatch_level), 0.35))
    support_floor = float(SUPPORT_FLOOR.get(str(support_regime), 0.03))
    behavior = (1.0 - mix) * target_probs + mix * anti
    behavior += 0.03 * rng.random(size=behavior.shape)
    behavior = _apply_support_stress(behavior, target_probs, support_regime=support_regime)
    behavior = _apply_support_floor(behavior, support_floor)

    return PolicyBundle(
        target_policy=TabularPolicy(target_probs if q.shape[0] > 1 else target_probs[0]),
        behavior_policy=TabularPolicy(behavior if q.shape[0] > 1 else behavior[0]),
        metadata={
            "mismatch_mix": mix,
            "support_floor": support_floor,
        },
    )
