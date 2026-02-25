#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

import numpy as np

from ess_ope.data.generate import generate_offline_dataset
from ess_ope.estimators.dm_fqe import direct_model_tabular, fitted_q_evaluation
from ess_ope.estimators.dr import doubly_robust_estimate
from ess_ope.estimators import is_family_estimates
from ess_ope.estimators.mrdr import mrdr_linear_estimate
from ess_ope.envs.random_mdp import RandomMDP, RandomMDPConfig
from ess_ope.evaluation.ground_truth import dynamic_programming_value
from ess_ope.metrics.ess import weight_summary
from ess_ope.policies.softmax import softmax_policy_from_logits, statewise_kl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single OPE diagnostics condition")
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--states", type=int, default=100)
    parser.add_argument("--actions", type=int, default=6)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    env = RandomMDP.generate(
        RandomMDPConfig(
            num_states=args.states,
            num_actions=args.actions,
            horizon=args.horizon,
            beta=args.beta,
            seed=args.seed,
        )
    )

    rng = np.random.default_rng(args.seed)
    theta_target = rng.normal(size=(env.num_states, env.num_actions))
    theta_random = rng.normal(size=(env.num_states, env.num_actions))
    theta_behavior = (1.0 - args.alpha) * theta_target + args.alpha * theta_random

    target = softmax_policy_from_logits(theta_target, temperature=args.temperature)
    behavior = softmax_policy_from_logits(theta_behavior, temperature=args.temperature)

    truth = dynamic_programming_value(env, target, gamma=args.gamma)

    dataset = generate_offline_dataset(
        env=env,
        behavior_policy=behavior,
        num_episodes=args.episodes,
        horizon=env.horizon,
        seed=args.seed + 123,
    )

    is_res = is_family_estimates(dataset, target, behavior, gamma=args.gamma)
    dm_res = direct_model_tabular(
        dataset=dataset,
        target_policy=target,
        num_states=env.num_states,
        num_actions=env.num_actions,
        horizon=env.horizon,
        initial_state_dist=env.initial_state_dist,
        gamma=args.gamma,
    )
    fqe_tab = fitted_q_evaluation(
        dataset=dataset,
        target_policy=target,
        num_states=env.num_states,
        num_actions=env.num_actions,
        horizon=env.horizon,
        initial_state_dist=env.initial_state_dist,
        model_type="tabular",
        gamma=args.gamma,
    )
    fqe_lin = fitted_q_evaluation(
        dataset=dataset,
        target_policy=target,
        num_states=env.num_states,
        num_actions=env.num_actions,
        horizon=env.horizon,
        initial_state_dist=env.initial_state_dist,
        model_type="linear",
        feature_tensor=env.linear_sa_features,
        gamma=args.gamma,
    )
    dr_val = doubly_robust_estimate(dataset, target, behavior, q_hat=fqe_lin.q, v_hat=fqe_lin.v, gamma=args.gamma)
    mrdr_res = mrdr_linear_estimate(
        dataset=dataset,
        target_policy=target,
        behavior_policy=behavior,
        feature_tensor=env.linear_sa_features,
        num_states=env.num_states,
        num_actions=env.num_actions,
        horizon=env.horizon,
        initial_state_dist=env.initial_state_dist,
        gamma=args.gamma,
    )

    summary = {
        "env_id": env.env_id,
        "alpha": args.alpha,
        "beta": args.beta,
        "episodes": args.episodes,
        "v_true": truth.value,
        "state_kl_mean": float(np.mean(statewise_kl(target, behavior))),
        **weight_summary(np.asarray(is_res["episode_weights"])),
        "estimate_is_trajectory": float(is_res["is_trajectory"]),
        "estimate_wis_trajectory": float(is_res["wis_trajectory"]),
        "estimate_is_pdis": float(is_res["is_pdis"]),
        "estimate_wis_pdis": float(is_res["wis_pdis"]),
        "estimate_dm_tabular": dm_res.value,
        "estimate_fqe_tabular": fqe_tab.value,
        "estimate_fqe_linear": fqe_lin.value,
        "estimate_dr": dr_val,
        "estimate_mrdr": mrdr_res.value,
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
