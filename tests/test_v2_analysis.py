from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ess_ope.v2.analysis import generate_phase_artifacts


def test_v2_analysis_tables_respect_native_ess_scope(tmp_path: Path) -> None:
    raw = pd.DataFrame(
        [
            {"phase_name": "p", "env_family": "contextual_bandit", "condition_id": "c", "replicate_id": 0, "estimator_key": "is", "ci_method": "point", "ci_level": np.nan, "estimate": 1.0, "true_value": 0.8, "error": 0.2, "abs_error": 0.2, "squared_error": 0.04, "native_diagnostic_kind": "ess", "native_diagnostic_value": 12.0, "seed": 0, "sample_size": 20, "mismatch_level": "low", "reward_noise_level": "low", "support_regime": "full", "horizon": 1, "reward_regime": "normal", "rarity_level": "none"},
            {"phase_name": "p", "env_family": "contextual_bandit", "condition_id": "c", "replicate_id": 1, "estimator_key": "is", "ci_method": "point", "ci_level": np.nan, "estimate": 0.9, "true_value": 0.8, "error": 0.1, "abs_error": 0.1, "squared_error": 0.01, "native_diagnostic_kind": "ess", "native_diagnostic_value": 10.0, "seed": 0, "sample_size": 20, "mismatch_level": "low", "reward_noise_level": "low", "support_regime": "full", "horizon": 1, "reward_regime": "normal", "rarity_level": "none"},
            {"phase_name": "p", "env_family": "contextual_bandit", "condition_id": "c", "replicate_id": 0, "estimator_key": "dr", "ci_method": "point", "ci_level": np.nan, "estimate": 0.82, "true_value": 0.8, "error": 0.02, "abs_error": 0.02, "squared_error": 0.0004, "native_diagnostic_kind": None, "native_diagnostic_value": np.nan, "seed": 0, "sample_size": 20, "mismatch_level": "low", "reward_noise_level": "low", "support_regime": "full", "horizon": 1, "reward_regime": "normal", "rarity_level": "none"},
            {"phase_name": "p", "env_family": "contextual_bandit", "condition_id": "c", "replicate_id": 1, "estimator_key": "dr", "ci_method": "point", "ci_level": np.nan, "estimate": 0.78, "true_value": 0.8, "error": -0.02, "abs_error": 0.02, "squared_error": 0.0004, "native_diagnostic_kind": None, "native_diagnostic_value": np.nan, "seed": 0, "sample_size": 20, "mismatch_level": "low", "reward_noise_level": "low", "support_regime": "full", "horizon": 1, "reward_regime": "normal", "rarity_level": "none"},
            {"phase_name": "p", "env_family": "contextual_bandit", "condition_id": "c", "replicate_id": 0, "estimator_key": "is", "ci_method": "bootstrap_percentile", "ci_level": 0.95, "estimate": 1.0, "true_value": 0.8, "error": 0.2, "abs_error": 0.2, "squared_error": 0.04, "ci_width": 0.3, "covered": 1.0, "native_diagnostic_kind": "ess", "native_diagnostic_value": 12.0, "seed": 0, "sample_size": 20, "mismatch_level": "low", "reward_noise_level": "low", "support_regime": "full", "horizon": 1, "reward_regime": "normal", "rarity_level": "none"},
            {"phase_name": "p", "env_family": "contextual_bandit", "condition_id": "c", "replicate_id": 1, "estimator_key": "is", "ci_method": "bootstrap_percentile", "ci_level": 0.95, "estimate": 0.9, "true_value": 0.8, "error": 0.1, "abs_error": 0.1, "squared_error": 0.01, "ci_width": 0.25, "covered": 1.0, "native_diagnostic_kind": "ess", "native_diagnostic_value": 10.0, "seed": 0, "sample_size": 20, "mismatch_level": "low", "reward_noise_level": "low", "support_regime": "full", "horizon": 1, "reward_regime": "normal", "rarity_level": "none"},
            {"phase_name": "p", "env_family": "contextual_bandit", "condition_id": "c", "replicate_id": 0, "estimator_key": "dr", "ci_method": "bootstrap_percentile", "ci_level": 0.95, "estimate": 0.82, "true_value": 0.8, "error": 0.02, "abs_error": 0.02, "squared_error": 0.0004, "ci_width": 0.1, "covered": 1.0, "native_diagnostic_kind": None, "native_diagnostic_value": np.nan, "seed": 0, "sample_size": 20, "mismatch_level": "low", "reward_noise_level": "low", "support_regime": "full", "horizon": 1, "reward_regime": "normal", "rarity_level": "none"},
            {"phase_name": "p", "env_family": "contextual_bandit", "condition_id": "c", "replicate_id": 1, "estimator_key": "dr", "ci_method": "bootstrap_percentile", "ci_level": 0.95, "estimate": 0.78, "true_value": 0.8, "error": -0.02, "abs_error": 0.02, "squared_error": 0.0004, "ci_width": 0.1, "covered": 1.0, "native_diagnostic_kind": None, "native_diagnostic_value": np.nan, "seed": 0, "sample_size": 20, "mismatch_level": "low", "reward_noise_level": "low", "support_regime": "full", "horizon": 1, "reward_regime": "normal", "rarity_level": "none"},
        ]
    )
    artifacts = generate_phase_artifacts(raw, tmp_path)
    assert not artifacts["table_a_estimator_summary"].empty
    assert not artifacts["table_b_diagnostic_quality"].empty
    dr_points = artifacts["point_estimates"][artifacts["point_estimates"]["estimator_key"] == "dr"]
    assert dr_points["native_diagnostic_value"].isna().all()
    quality = artifacts["table_b_diagnostic_quality"].set_index("estimator_key")
    assert np.isnan(quality.loc["dr", "corr_ess_abs_error"])
    assert (tmp_path / "table_a_estimator_summary.csv").exists()
