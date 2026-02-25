from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ess_ope.envs.base import DiscreteFiniteHorizonEnv


@dataclass
class GridworldConfig:
    width: int = 5
    height: int = 5
    horizon: int = 20
    slip_prob: float = 0.05
    seed: int = 0


class Gridworld(DiscreteFiniteHorizonEnv):
    """Simple stochastic gridworld for interpretability experiments."""

    ACTIONS = {
        0: (-1, 0),  # up
        1: (1, 0),   # down
        2: (0, -1),  # left
        3: (0, 1),   # right
    }

    @classmethod
    def generate(cls, config: GridworldConfig) -> "Gridworld":
        rng = np.random.default_rng(config.seed)

        n_states = config.width * config.height
        n_actions = len(cls.ACTIONS)

        def to_state(r: int, c: int) -> int:
            return r * config.width + c

        def to_pos(state: int) -> Tuple[int, int]:
            return divmod(state, config.width)

        goal_state = to_state(config.height - 1, config.width - 1)

        transition_sa = np.zeros((n_states, n_actions, n_states), dtype=float)
        rewards_sa = np.full((n_states, n_actions), -0.02, dtype=float)

        for s in range(n_states):
            r, c = to_pos(s)
            for a, (dr, dc) in cls.ACTIONS.items():
                for alt_a, (adr, adc) in cls.ACTIONS.items():
                    p = (1.0 - config.slip_prob) if alt_a == a else config.slip_prob / (n_actions - 1)
                    nr = np.clip(r + adr, 0, config.height - 1)
                    nc = np.clip(c + adc, 0, config.width - 1)
                    ns = to_state(int(nr), int(nc))
                    transition_sa[s, a, ns] += p

                if s == goal_state:
                    rewards_sa[s, a] = 0.0
                elif np.argmax(transition_sa[s, a]) == goal_state:
                    rewards_sa[s, a] = 1.0

        transition_probs = np.repeat(transition_sa[None, ...], config.horizon, axis=0)
        rewards = np.repeat(rewards_sa[None, ...], config.horizon, axis=0)

        initial_state_dist = np.zeros(n_states, dtype=float)
        initial_state_dist[to_state(0, 0)] = 1.0

        env = cls(
            transition_probs=transition_probs,
            rewards=rewards,
            initial_state_dist=initial_state_dist,
            horizon=config.horizon,
            seed=config.seed,
        )
        env.env_id = f"gridworld_{config.width}x{config.height}_h{config.horizon}_seed{config.seed}"
        return env
