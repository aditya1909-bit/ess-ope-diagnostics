from __future__ import annotations

from pathlib import Path

from ess_ope.utils.logging import refresh_latest_pointer, resolve_latest_path, sync_tracked_latest_snapshot


def test_refresh_latest_pointer_writes_manifest_and_env_aliases(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    results_root.mkdir()
    (results_root / "20260325_140042_random_mdp_ci_ultra").mkdir()
    (results_root / "20260325_142812_chain_bandit_ci_focused").mkdir()

    refresh_latest_pointer(results_root)

    manifest_path = results_root / "latest_runs.json"
    assert manifest_path.exists()
    assert resolve_latest_path(results_root / "latest_random_mdp" / "sweep_results.parquet") == (
        results_root / "20260325_140042_random_mdp_ci_ultra" / "sweep_results.parquet"
    )
    assert resolve_latest_path(results_root / "latest_chain_bandit" / "figures") == (
        results_root / "20260325_142812_chain_bandit_ci_focused" / "figures"
    )


def test_sync_tracked_latest_snapshot_materializes_git_tracked_dir(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    run_dir = results_root / "20260325_140042_random_mdp_ci_ultra"
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text("name: random_mdp_ci_ultra\n", encoding="utf-8")
    (run_dir / "metadata.json").write_text('{"git_commit":"abc"}\n', encoding="utf-8")
    (run_dir / "sweep_results.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    (figures_dir / "benchmark_report.csv").write_text("metric,value\nm,1\n", encoding="utf-8")

    snapshot_dir = sync_tracked_latest_snapshot(results_root, run_dir, "random_mdp")

    assert snapshot_dir == results_root / "latest_random_mdp"
    assert (snapshot_dir / "sweep_results.csv").exists()
    assert (snapshot_dir / "figures" / "benchmark_report.csv").exists()
    assert resolve_latest_path(snapshot_dir / "sweep_results.csv") == snapshot_dir / "sweep_results.csv"
