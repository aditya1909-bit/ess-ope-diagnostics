from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class EnvStep:
    state: int
    action: int
    reward: float
    next_state: int
    done: bool


class DiscreteFiniteHorizonEnv:
    """Finite-horizon discrete MDP with known transition and reward tensors."""

    def __init__(
        self,
        transition_probs: np.ndarray,
        rewards: np.ndarray,
        initial_state_dist: np.ndarray,
        horizon: int,
        seed: Optional[int] = None,
    ) -> None:
        transition_probs = np.asarray(transition_probs, dtype=float)
        rewards = np.asarray(rewards, dtype=float)
        initial_state_dist = np.asarray(initial_state_dist, dtype=float)

        if transition_probs.ndim != 4:
            raise ValueError("transition_probs must have shape (H, S, A, S)")
        if rewards.ndim != 3:
            raise ValueError("rewards must have shape (H, S, A)")

        self.horizon = int(horizon)
        self.transition_probs = transition_probs
        self.rewards = rewards
        self.initial_state_dist = initial_state_dist / np.sum(initial_state_dist)

        self.num_states = transition_probs.shape[1]
        self.num_actions = transition_probs.shape[2]

        if transition_probs.shape[0] != self.horizon:
            raise ValueError("Transition horizon mismatch")
        if rewards.shape[0] != self.horizon:
            raise ValueError("Reward horizon mismatch")
        if rewards.shape[1] != self.num_states or rewards.shape[2] != self.num_actions:
            raise ValueError("Reward shape mismatch")
        if initial_state_dist.shape != (self.num_states,):
            raise ValueError("initial_state_dist must have shape (S,)")

        row_sums = transition_probs.sum(axis=-1)
        if not np.allclose(row_sums, 1.0, atol=1e-6):
            raise ValueError("Transition probabilities must sum to 1 along next-state axis")

        self._rng = np.random.default_rng(seed)
        self._t = 0
        self._state = 0
        self._done = False

    def reset(self, seed: Optional[int] = None) -> int:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        self._done = False
        self._state = int(self._rng.choice(self.num_states, p=self.initial_state_dist))
        return self._state

    def step(self, action: int) -> Tuple[int, float, bool, Dict[str, int]]:
        if self._done:
            raise RuntimeError("Episode already finished; call reset().")
        if not 0 <= action < self.num_actions:
            raise ValueError("Action out of bounds")

        probs = self.transition_probs[self._t, self._state, action]
        next_state = int(self._rng.choice(self.num_states, p=probs))
        reward = float(self.rewards[self._t, self._state, action])

        self._t += 1
        done = self._t >= self.horizon
        self._done = done

        prev_state = self._state
        self._state = next_state
        info = {"t": self._t, "prev_state": prev_state}
        return next_state, reward, done, info

    def expected_reward(self, t: int, s: int, a: int) -> float:
        return float(self.rewards[t, s, a])

    def copy_with_seed(self, seed: Optional[int]) -> "DiscreteFiniteHorizonEnv":
        return DiscreteFiniteHorizonEnv(
            transition_probs=self.transition_probs.copy(),
            rewards=self.rewards.copy(),
            initial_state_dist=self.initial_state_dist.copy(),
            horizon=self.horizon,
            seed=seed,
        )
