from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class TabularPolicy:
    """Tabular policy supporting stationary (S,A) or time-varying (H,S,A) probs."""

    probs: np.ndarray

    def __post_init__(self) -> None:
        p = np.asarray(self.probs, dtype=float)
        if p.ndim not in (2, 3):
            raise ValueError("Policy probabilities must have shape (S,A) or (H,S,A)")
        row_sums = p.sum(axis=-1)
        if not np.allclose(row_sums, 1.0, atol=1e-6):
            raise ValueError("Policy probabilities must sum to 1 along action axis")
        if np.any(p < 0):
            raise ValueError("Policy probabilities must be non-negative")
        self.probs = p

    @property
    def num_states(self) -> int:
        return self.probs.shape[-2]

    @property
    def num_actions(self) -> int:
        return self.probs.shape[-1]

    def probs_at(self, state: int, t: int = 0) -> np.ndarray:
        if self.probs.ndim == 2:
            return self.probs[state]
        return self.probs[t, state]

    def prob(self, state: int, action: int, t: int = 0) -> float:
        return float(self.probs_at(state, t)[action])

    def sample_action(self, state: int, t: int = 0, rng: Optional[np.random.Generator] = None) -> int:
        generator = rng if rng is not None else np.random.default_rng()
        return int(generator.choice(self.num_actions, p=self.probs_at(state, t)))

    @classmethod
    def uniform(cls, num_states: int, num_actions: int, horizon: Optional[int] = None) -> "TabularPolicy":
        if horizon is None:
            probs = np.full((num_states, num_actions), 1.0 / num_actions, dtype=float)
        else:
            probs = np.full((horizon, num_states, num_actions), 1.0 / num_actions, dtype=float)
        return cls(probs)
