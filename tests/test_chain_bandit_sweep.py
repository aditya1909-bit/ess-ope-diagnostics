from __future__ import annotations

from ess_ope.evaluation.sweep import SweepConfig, _evaluate_condition, _iter_tasks


def test_iter_tasks_expands_chain_axes() -> None:
    cfg = SweepConfig(
        env_name="chain_bandit",
        seeds=[0],
        alphas=[0.1, 0.4],
        betas=[0.0, 0.5],
        dataset_sizes=[50],
        transition_strengths=[0.0, 0.5],
        reward_mean_scales=[1.0],
        reward_gaps=[0.2, 0.6],
        reward_stds=[0.1, 1.0],
        chain_variants=["reward_only", "transitional"],
    )

    tasks = list(_iter_tasks(cfg))
    expected = (
        len(cfg.seeds)
        * len(cfg.betas)
        * len(cfg.transition_strengths)
        * len(cfg.reward_mean_scales)
        * len(cfg.reward_gaps)
        * len(cfg.reward_stds)
        * len(cfg.chain_variants)
        * len(cfg.alphas)
        * len(cfg.dataset_sizes)
        * int(cfg.env_repeats)
        * int(cfg.policy_repeats)
        * int(cfg.dataset_repeats)
    )
    assert len(tasks) == expected


def test_single_chain_condition_evaluates() -> None:
    cfg = SweepConfig(
        env_name="chain_bandit",
        seeds=[0],
        alphas=[0.2],
        betas=[0.1],
        dataset_sizes=[40],
        transition_strengths=[0.7],
        reward_mean_scales=[1.0],
        reward_gaps=[0.5],
        reward_stds=[0.2],
        chain_variants=["transitional"],
        env={"num_states": 3, "num_actions": 3, "horizon": 4, "linear_feature_dim": 8},
    )

    task = next(_iter_tasks(cfg))
    rows = _evaluate_condition(task)
    assert len(rows) == 1

    row = rows[0]
    assert row["env_name"] == "chain_bandit"
    assert row["transition_strength"] == 0.7
    assert row["reward_mean_scale"] == 1.0
    assert row["reward_gap"] == 0.5
    assert row["reward_std"] == 0.2
    assert row["chain_variant"] == "transitional"
