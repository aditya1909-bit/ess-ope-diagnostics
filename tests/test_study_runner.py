from __future__ import annotations

from pathlib import Path

from ess_ope.study.config import StudyConfig
from ess_ope.study.runner import run_study


def test_tiny_study_smoke_produces_all_paper_artifacts(tmp_path: Path) -> None:
    cfg = StudyConfig(
        name="paper_tiny_smoke",
        results_root=str(tmp_path),
        output_subdir="artifacts",
        experiment_configs=[
            "configs/study/experiment_1.yaml",
            "configs/study/experiment_2.yaml",
            "configs/study/experiment_3.yaml",
            "configs/study/experiment_4.yaml",
            "configs/study/experiment_5.yaml",
        ],
        overrides={
            "replicates": 2,
            "intervals": {"bootstrap_samples": 4, "primary_level": 0.9},
            "grid": {"sample_size": [20, 40]},
        },
    )
    raw_df, run_dir, artifacts = run_study(cfg)
    assert not raw_df.empty
    assert not artifacts["table_1_main_summary"].empty
    assert not artifacts["table_2_diagnostic_usefulness"].empty
    for idx in range(1, 9):
        assert (run_dir / "artifacts" / f"figure_{idx}.png").exists()
    assert (run_dir / "artifacts" / "replicate_results.csv").exists()
    assert (run_dir / "artifacts" / "condition_summary.csv").exists()
