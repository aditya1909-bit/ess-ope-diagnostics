from __future__ import annotations

from pathlib import Path

from ess_ope.evaluation.summary import (
    build_bias_variance_summary,
    build_ci_coverage_summary,
    build_ci_interval_summary,
    build_diagnostic_comparability_summary,
)
from ess_ope.evaluation.sweep import SweepConfig, run_sweep
from ess_ope.plotting.benchmark_figures import generate_benchmark_figures


def test_interval_pipeline_outputs_new_columns_and_figures(tmp_path: Path) -> None:
    config = SweepConfig(
        name="tiny_interval_pipeline",
        results_root=str(tmp_path),
        env_name="random_mdp",
        seeds=[0],
        alphas=[0.2, 0.6],
        betas=[0.0, 0.5],
        dataset_sizes=[10],
        env_repeats=1,
        policy_repeats=1,
        dataset_repeats=2,
        num_workers=1,
        analysis_estimators=["is_pdis", "dr_oracle", "dm_tabular", "fqe_linear"],
        interval_mode="both",
        ci_level=0.95,
        ci_bootstrap_samples=4,
        env={
            "num_states": 8,
            "num_actions": 3,
            "horizon": 4,
            "branch_factor": 2,
            "linear_feature_dim": 5,
            "reward_noise_std": 0.0,
        },
    )
    df, _, _ = run_sweep(config)

    expected_cols = {
        "estimate_dr_oracle",
        "ci_analytic_low_is_pdis",
        "ci_analytic_low_dr_oracle",
        "ci_bootstrap_low_dm_tabular",
        "ci_bootstrap_low_fqe_linear",
    }
    assert expected_cols.issubset(df.columns)

    bias_variance_summary = build_bias_variance_summary(df, estimator_keys=config.analysis_estimators)
    ci_interval_summary = build_ci_interval_summary(df, estimator_keys=config.analysis_estimators, methods=["analytic", "bootstrap"])
    ci_coverage_summary = build_ci_coverage_summary(ci_interval_summary, ci_level=config.ci_level)
    diagnostic_comparability = build_diagnostic_comparability_summary(
        df,
        ci_coverage_summary=ci_coverage_summary,
        estimator_keys=config.analysis_estimators,
    )

    assert not bias_variance_summary.empty
    assert not ci_interval_summary.empty
    assert not ci_coverage_summary.empty
    assert not diagnostic_comparability.empty
    assert {"DR", "DM", "FQE", "IS-PDIS"}.issubset(set(ci_interval_summary["estimator"]))

    fig_dir = tmp_path / "figures"
    report = generate_benchmark_figures(
        df=df,
        output_dir=fig_dir,
        bias_variance_summary=bias_variance_summary,
        ci_interval_summary=ci_interval_summary,
        ci_coverage_summary=ci_coverage_summary,
    )
    assert not report.empty
    for stem in [
        "benchmark_fig6_ess_bias_variance_mse",
        "benchmark_fig7_matched_ess_risk_curves",
        "benchmark_fig8_ci_coverage_width",
        "benchmark_fig9_ci_method_comparison",
    ]:
        assert (fig_dir / f"{stem}.png").exists()
