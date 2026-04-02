from __future__ import annotations

from pathlib import Path

from ess_ope.v2.config import PhaseConfig, SuiteConfig
from ess_ope.v2.runner import run_phase, run_suite


def test_v2_phase_and_suite_smoke(tmp_path: Path) -> None:
    bandit_cfg = PhaseConfig(
        name="v2_smoke_bandit",
        env_family="contextual_bandit",
        results_root=str(tmp_path),
        seeds=[0],
        replicates=2,
        sample_sizes=[12],
        mismatch_levels=["low"],
        reward_noise_levels=["low"],
        support_regimes=["full"],
        horizons=[1],
        reward_regimes=["normal"],
        rarity_levels=["none"],
        estimators=["is", "dr"],
        ci_methods=["analytic", "bootstrap_percentile"],
        ci_levels=[0.9],
        bootstrap_samples=2,
        include_reference_estimators=True,
        env={"num_contexts": 4, "num_actions": 2, "linear_feature_dim": 6, "fqe_l2_reg": 1e-4},
    )
    raw_df, run_dir, artifacts = run_phase(bandit_cfg)
    assert not raw_df.empty
    assert not artifacts["table_a_estimator_summary"].empty
    assert (run_dir / "artifacts" / "table_b_diagnostic_quality.csv").exists()
    assert any((run_dir / "artifacts").glob("v2_fig*.png"))

    tabular_cfg = PhaseConfig(
        name="v2_smoke_tabular",
        env_family="tabular_mdp",
        results_root=str(tmp_path),
        seeds=[0],
        replicates=1,
        sample_sizes=[10],
        mismatch_levels=["medium"],
        reward_noise_levels=["low"],
        support_regimes=["weak"],
        horizons=[3],
        reward_regimes=["normal"],
        rarity_levels=["none"],
        estimators=["pdis", "wdr", "fqe_linear"],
        ci_methods=["analytic", "bootstrap_percentile"],
        ci_levels=[0.9],
        bootstrap_samples=2,
        include_reference_estimators=False,
        env={"num_states": 6, "num_actions": 2, "branch_factor": 2, "linear_feature_dim": 6, "fqe_l2_reg": 1e-4},
    )
    bandit_path = tmp_path / "bandit.yaml"
    tabular_path = tmp_path / "tabular.yaml"
    from ess_ope.utils.config import dump_yaml

    dump_yaml(bandit_path, bandit_cfg.to_dict())
    dump_yaml(tabular_path, tabular_cfg.to_dict())

    suite_cfg = SuiteConfig(
        name="v2_smoke_suite",
        results_root=str(tmp_path),
        output_subdir="paper_artifacts",
        phase_configs=[str(bandit_path), str(tabular_path)],
    )
    combined, suite_dir = run_suite(suite_cfg)
    assert not combined.empty
    assert (suite_dir / "paper_artifacts" / "table_a_estimator_summary.csv").exists()
    assert any((suite_dir / "paper_artifacts").glob("v2_fig*.png"))
