from __future__ import annotations

import numpy as np

from ess_ope.envs.base import DiscreteFiniteHorizonEnv


def with_reward_scale(env: DiscreteFiniteHorizonEnv, scale: float) -> DiscreteFiniteHorizonEnv:
    """Return a copy of env with rewards multiplied by scale."""
    return DiscreteFiniteHorizonEnv(
        transition_probs=env.transition_probs.copy(),
        rewards=np.asarray(env.rewards * scale, dtype=float),
        initial_state_dist=env.initial_state_dist.copy(),
        horizon=env.horizon,
    )
