from __future__ import annotations

from typing import Optional

import numpy as np

from ess_ope.data.dataset import EpisodeDataset
from ess_ope.envs.base import DiscreteFiniteHorizonEnv
from ess_ope.policies.tabular import TabularPolicy


def generate_offline_dataset(
    env: DiscreteFiniteHorizonEnv,
    behavior_policy: TabularPolicy,
    num_episodes: int,
    horizon: Optional[int] = None,
    seed: int = 0,
) -> EpisodeDataset:
    """Generate offline episodes with logged transitions from behavior policy."""
    h = env.horizon if horizon is None else int(horizon)
    if h > env.horizon:
        raise ValueError("Requested horizon exceeds environment horizon")

    rng = np.random.default_rng(seed)

    states = np.zeros((num_episodes, h), dtype=int)
    actions = np.zeros((num_episodes, h), dtype=int)
    rewards = np.zeros((num_episodes, h), dtype=float)
    next_states = np.zeros((num_episodes, h), dtype=int)
    dones = np.zeros((num_episodes, h), dtype=bool)

    for ep in range(num_episodes):
        s = env.reset(seed=int(rng.integers(0, 2**32 - 1)))
        for t in range(h):
            a = behavior_policy.sample_action(s, t=t, rng=rng)
            sp, r, done, _ = env.step(a)

            states[ep, t] = s
            actions[ep, t] = a
            rewards[ep, t] = r
            next_states[ep, t] = sp
            dones[ep, t] = done

            s = sp
            if done and t < h - 1:
                states[ep, t + 1 :] = s
                actions[ep, t + 1 :] = 0
                rewards[ep, t + 1 :] = 0.0
                next_states[ep, t + 1 :] = s
                dones[ep, t + 1 :] = True
                break

    return EpisodeDataset(
        states=states,
        actions=actions,
        rewards=rewards,
        next_states=next_states,
        dones=dones,
    )
