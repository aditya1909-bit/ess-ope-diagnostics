from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Iterator, Tuple

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class EpisodeDataset:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray

    def __post_init__(self) -> None:
        self.states = np.asarray(self.states, dtype=int)
        self.actions = np.asarray(self.actions, dtype=int)
        self.rewards = np.asarray(self.rewards, dtype=float)
        self.next_states = np.asarray(self.next_states, dtype=int)
        self.dones = np.asarray(self.dones, dtype=bool)

        shape = self.states.shape
        for name, arr in (
            ("actions", self.actions),
            ("rewards", self.rewards),
            ("next_states", self.next_states),
            ("dones", self.dones),
        ):
            if arr.shape != shape:
                raise ValueError(f"{name} shape {arr.shape} does not match states shape {shape}")

        if self.states.ndim != 2:
            raise ValueError("Dataset arrays must have shape (K, H)")

    @property
    def num_episodes(self) -> int:
        return int(self.states.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.states.shape[1])

    def episodes(self) -> Iterator[Dict[str, np.ndarray]]:
        for k in range(self.num_episodes):
            yield {
                "states": self.states[k],
                "actions": self.actions[k],
                "rewards": self.rewards[k],
                "next_states": self.next_states[k],
                "dones": self.dones[k],
            }

    def transitions_at_time(self, t: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            self.states[:, t],
            self.actions[:, t],
            self.rewards[:, t],
            self.next_states[:, t],
            self.dones[:, t],
        )

    def initial_states(self) -> np.ndarray:
        return self.states[:, 0]

    def to_long_dataframe(self) -> "pd.DataFrame":
        import pandas as pd

        k, h = self.states.shape
        episode_idx = np.repeat(np.arange(k), h)
        t = np.tile(np.arange(h), k)
        return pd.DataFrame(
            {
                "episode": episode_idx,
                "t": t,
                "s": self.states.reshape(-1),
                "a": self.actions.reshape(-1),
                "r": self.rewards.reshape(-1),
                "sp": self.next_states.reshape(-1),
                "done": self.dones.reshape(-1),
            }
        )
