from __future__ import annotations

from pathlib import Path

from ess_ope.utils.logging import refresh_latest_pointer, resolve_latest_path


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
