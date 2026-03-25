from __future__ import annotations

import numpy as np

from ess_ope.data.generate import generate_offline_dataset
from ess_ope.estimators.dm_fqe import fitted_q_evaluation
from ess_ope.envs.chain_bandit import ChainBanditConfig, ChainBanditEnv
from ess_ope.policies.tabular import TabularPolicy


def test_linear_fqe_accepts_time_varying_chain_bandit_features() -> None:
    env = ChainBanditEnv.generate(
        ChainBanditConfig(
            num_states=8,
            num_actions=3,
            horizon=6,
            linear_feature_dim=12,
            seed=0,
        )
    )
    behavior = TabularPolicy.uniform(env.num_states, env.num_actions, horizon=env.horizon)
    target = TabularPolicy.uniform(env.num_states, env.num_actions, horizon=env.horizon)
    dataset = generate_offline_dataset(
        env=env,
        behavior_policy=behavior,
        num_episodes=16,
        horizon=env.horizon,
        seed=1,
    )

    result = fitted_q_evaluation(
        dataset=dataset,
        target_policy=target,
        num_states=env.num_states,
        num_actions=env.num_actions,
        horizon=env.horizon,
        initial_state_dist=env.initial_state_dist,
        model_type="linear",
        feature_tensor=env.linear_sa_features,
        l2_reg=1e-4,
    )

    assert np.isfinite(result.value)
    assert result.q.shape == (env.horizon, env.num_states, env.num_actions)
