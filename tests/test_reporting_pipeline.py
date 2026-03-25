from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from ess_ope.evaluation.sweep import SweepConfig
from ess_ope.plotting import benchmark_figures
from ess_ope.utils.config import dump_yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_generate_benchmark_figures_respects_nondefault_ci_level(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, float] = {}

    def fake_figure_8(ci_coverage_summary: pd.DataFrame, output_dir: str | Path, ci_level: float = 0.95) -> None:
        captured["figure_8"] = ci_level

    def fake_figure_11(
        ci_coverage_summary: pd.DataFrame,
        bias_variance_summary: pd.DataFrame,
        output_dir: str | Path,
        ci_level: float = 0.95,
    ) -> None:
        captured["figure_11"] = ci_level

    monkeypatch.setattr(benchmark_figures, "figure_1_ess_vs_error_by_estimator", lambda *args, **kwargs: None)
    monkeypatch.setattr(benchmark_figures, "figure_2_same_ess_different_error", lambda *args, **kwargs: 0.2)
    monkeypatch.setattr(benchmark_figures, "figure_3_ess_changes_error_stability", lambda *args, **kwargs: None)
    monkeypatch.setattr(benchmark_figures, "figure_4_fan_estimate_vs_ess", lambda *args, **kwargs: {})
    monkeypatch.setattr(benchmark_figures, "figure_5_mean_variance_sensitivity", lambda *args, **kwargs: None)
    monkeypatch.setattr(benchmark_figures, "figure_6_ess_bias_variance_mse", lambda *args, **kwargs: None)
    monkeypatch.setattr(benchmark_figures, "figure_7_matched_ess_risk_curves", lambda *args, **kwargs: None)
    monkeypatch.setattr(benchmark_figures, "figure_8_ci_coverage_width", fake_figure_8)
    monkeypatch.setattr(benchmark_figures, "figure_9_ci_method_comparison", lambda *args, **kwargs: None)
    monkeypatch.setattr(benchmark_figures, "figure_10_ess_story_dashboard", lambda *args, **kwargs: None)
    monkeypatch.setattr(benchmark_figures, "figure_11_ci_story_dashboard", fake_figure_11)

    df = pd.DataFrame(
        {
            "alpha": [0.2, 0.6],
            "beta": [0.0, 0.0],
            "K": [10, 10],
            "ess_is": [5.0, 8.0],
            "ess_is_over_k": [0.5, 0.8],
            "estimate_is_pdis": [0.0, 0.0],
            "error_is_pdis": [0.1, 0.2],
            "abs_error_is_pdis": [0.1, 0.2],
            "squared_error_is_pdis": [0.01, 0.04],
        }
    )
    bias_variance_summary = pd.DataFrame({"estimator": ["IS-PDIS"], "median_ess_norm": [0.5], "abs_bias": [0.1], "variance": [0.2], "mse": [0.3], "mean_abs_error": [0.1]})
    ci_interval_summary = pd.DataFrame(
        {
            "estimator": ["IS-PDIS"],
            "ci_method": ["bootstrap"],
            "estimate": [0.0],
            "error": [0.1],
            "abs_error": [0.1],
            "ci_width": [0.2],
            "covered": [1.0],
            "v_true": [0.0],
        }
    )
    ci_coverage_summary = pd.DataFrame(
        {
            "estimator": ["IS-PDIS"],
            "ci_method": ["bootstrap"],
            "coverage_rate": [0.8],
            "coverage_gap": [0.0],
            "mean_width": [0.2],
            "mean_half_width": [0.1],
            "mean_abs_error": [0.1],
            "mean_error": [0.05],
            "bias_to_half_width_ratio": [0.5],
        }
    )
    estimator_summary = pd.DataFrame(
        {
            "estimator": ["is_pdis"],
            "spearman_ess_abs_error": [-0.4],
            "spearman_ci_low": [-0.5],
            "spearman_ci_high": [-0.3],
        }
    )
    diagnostic_comparability = pd.DataFrame(
        {
            "diagnostic": ["ess"],
            "estimator": ["IS-PDIS"],
            "mean_ess_norm": [0.5],
            "mean_abs_error": [0.1],
            "mean_squared_error": [0.01],
            "ess_bin_low": [0.4],
            "ess_bin_high": [0.6],
            "cross_estimator_abs_error_spread": [0.0],
            "cross_estimator_mse_spread": [0.0],
        }
    )

    benchmark_figures.generate_benchmark_figures(
        df=df,
        output_dir=tmp_path,
        ci_level=0.8,
        bias_variance_summary=bias_variance_summary,
        ci_interval_summary=ci_interval_summary,
        ci_coverage_summary=ci_coverage_summary,
        estimator_summary=estimator_summary,
        diagnostic_comparability=diagnostic_comparability,
    )

    assert captured == {"figure_8": 0.8, "figure_11": 0.8}


def test_run_and_analyze_scripts_emit_same_summary_artifacts(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    reanalyze_dir = tmp_path / "reanalyzed" / "figures"
    config_path = tmp_path / "chain_tiny.yaml"
    dump_yaml(
        config_path,
        SweepConfig(
            name="chain_tiny",
            results_root=str(results_root),
            env_name="chain_bandit",
            seeds=[0],
            alphas=[0.2, 0.6],
            betas=[0.0],
            dataset_sizes=[12],
            transition_strengths=[0.5],
            reward_mean_scales=[1.0],
            reward_gaps=[0.4],
            reward_stds=[0.1],
            chain_variants=["transitional"],
            env_repeats=1,
            policy_repeats=1,
            dataset_repeats=1,
            num_workers=1,
            interval_mode="both",
            ci_level=0.8,
            ci_bootstrap_samples=2,
            env={"num_states": 3, "num_actions": 2, "horizon": 4, "linear_feature_dim": 8},
        ).to_dict(),
    )

    env = os.environ.copy()
    env.pop("MPLBACKEND", None)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")

    subprocess.run(
        [
            sys.executable,
            "experiments/run_benchmark.py",
            "--config",
            str(config_path),
            "--interval-mode",
            "both",
            "--ci-level",
            "0.8",
            "--ci-bootstrap-samples",
            "2",
            "--bootstrap-samples",
            "24",
            "--bootstrap-max-points",
            "200",
            "--num-workers",
            "1",
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    run_dirs = sorted(results_root.glob("20*_chain_tiny"))
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    subprocess.run(
        [
            sys.executable,
            "experiments/analyze_results.py",
            "--results",
            str(run_dir / "sweep_results.csv"),
            "--output-dir",
            str(reanalyze_dir),
            "--interval-mode",
            "both",
            "--ci-level",
            "0.8",
            "--bootstrap-samples",
            "24",
            "--bootstrap-max-points",
            "200",
            "--no-progress",
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    run_csvs = {path.name for path in (run_dir / "figures").glob("*.csv")}
    analyze_csvs = {path.name for path in reanalyze_dir.glob("*.csv")}
    assert run_csvs == analyze_csvs

    for name in ["benchmark_report.csv", "ci_coverage_summary.csv", "trial_scorecard.csv"]:
        run_df = pd.read_csv(run_dir / "figures" / name)
        analyze_df = pd.read_csv(reanalyze_dir / name)
        assert list(run_df.columns) == list(analyze_df.columns)

    trial_index = pd.read_csv(results_root / "trial_index.csv")
    assert "run_id" in trial_index.columns
    assert "fqe_bootstrap_coverage_rate" in trial_index.columns
    assert (results_root / "latest_chain_bandit" / "figures" / "trial_scorecard.csv").exists()
