from __future__ import annotations

import numpy as np
import pandas as pd

from ess_ope.study.analysis import generate_study_artifacts


def test_study_analysis_generates_main_tables(tmp_path) -> None:
    raw = pd.DataFrame(
        [
            {"experiment_id": "experiment_3", "experiment_title": "x", "env_name": "short_tabular_mdp", "condition_id": "c", "replicate_id": 0, "dataset_seed": 1, "estimator_key": "pdis", "estimator_label": "PDIS", "estimator_family": "is_like", "ci_method": "point", "ci_level": np.nan, "estimate": 1.0, "true_value": 0.8, "error": 0.2, "abs_error": 0.2, "squared_error": 0.04, "shared_wess": 12.0, "wess_native_applicable": True, "ci_low": np.nan, "ci_high": np.nan, "ci_width": np.nan, "covered": np.nan, "sample_size": 300, "mismatch_alpha": 0.4, "seed": 0},
            {"experiment_id": "experiment_3", "experiment_title": "x", "env_name": "short_tabular_mdp", "condition_id": "c", "replicate_id": 1, "dataset_seed": 2, "estimator_key": "pdis", "estimator_label": "PDIS", "estimator_family": "is_like", "ci_method": "point", "ci_level": np.nan, "estimate": 0.9, "true_value": 0.8, "error": 0.1, "abs_error": 0.1, "squared_error": 0.01, "shared_wess": 10.0, "wess_native_applicable": True, "ci_low": np.nan, "ci_high": np.nan, "ci_width": np.nan, "covered": np.nan, "sample_size": 300, "mismatch_alpha": 0.4, "seed": 0},
            {"experiment_id": "experiment_3", "experiment_title": "x", "env_name": "short_tabular_mdp", "condition_id": "off_slice", "replicate_id": 0, "dataset_seed": 10, "estimator_key": "pdis", "estimator_label": "PDIS", "estimator_family": "is_like", "ci_method": "point", "ci_level": np.nan, "estimate": 1.8, "true_value": 0.8, "error": 1.0, "abs_error": 1.0, "squared_error": 1.0, "shared_wess": 2.0, "wess_native_applicable": True, "ci_low": np.nan, "ci_high": np.nan, "ci_width": np.nan, "covered": np.nan, "sample_size": 100, "mismatch_alpha": 0.8, "seed": 0},
            {"experiment_id": "experiment_3", "experiment_title": "x", "env_name": "short_tabular_mdp", "condition_id": "c", "replicate_id": 0, "dataset_seed": 1, "estimator_key": "dr", "estimator_label": "DR", "estimator_family": "doubly_robust", "ci_method": "point", "ci_level": np.nan, "estimate": 0.82, "true_value": 0.8, "error": 0.02, "abs_error": 0.02, "squared_error": 0.0004, "shared_wess": 12.0, "wess_native_applicable": False, "ci_low": np.nan, "ci_high": np.nan, "ci_width": np.nan, "covered": np.nan, "sample_size": 300, "mismatch_alpha": 0.4, "seed": 0},
            {"experiment_id": "experiment_3", "experiment_title": "x", "env_name": "short_tabular_mdp", "condition_id": "c", "replicate_id": 1, "dataset_seed": 2, "estimator_key": "dr", "estimator_label": "DR", "estimator_family": "doubly_robust", "ci_method": "point", "ci_level": np.nan, "estimate": 0.78, "true_value": 0.8, "error": -0.02, "abs_error": 0.02, "squared_error": 0.0004, "shared_wess": 10.0, "wess_native_applicable": False, "ci_low": np.nan, "ci_high": np.nan, "ci_width": np.nan, "covered": np.nan, "sample_size": 300, "mismatch_alpha": 0.4, "seed": 0},
            {"experiment_id": "experiment_3", "experiment_title": "x", "env_name": "short_tabular_mdp", "condition_id": "c", "replicate_id": 0, "dataset_seed": 1, "estimator_key": "pdis", "estimator_label": "PDIS", "estimator_family": "is_like", "ci_method": "bootstrap_percentile", "ci_level": 0.9, "estimate": 1.0, "true_value": 0.8, "error": 0.2, "abs_error": 0.2, "squared_error": 0.04, "shared_wess": 12.0, "wess_native_applicable": True, "ci_low": 0.7, "ci_high": 1.3, "ci_width": 0.6, "covered": 1.0, "sample_size": 300, "mismatch_alpha": 0.4, "seed": 0},
            {"experiment_id": "experiment_3", "experiment_title": "x", "env_name": "short_tabular_mdp", "condition_id": "c", "replicate_id": 1, "dataset_seed": 2, "estimator_key": "pdis", "estimator_label": "PDIS", "estimator_family": "is_like", "ci_method": "bootstrap_percentile", "ci_level": 0.9, "estimate": 0.9, "true_value": 0.8, "error": 0.1, "abs_error": 0.1, "squared_error": 0.01, "shared_wess": 10.0, "wess_native_applicable": True, "ci_low": 0.6, "ci_high": 1.2, "ci_width": 0.6, "covered": 1.0, "sample_size": 300, "mismatch_alpha": 0.4, "seed": 0},
            {"experiment_id": "experiment_3", "experiment_title": "x", "env_name": "short_tabular_mdp", "condition_id": "off_slice", "replicate_id": 0, "dataset_seed": 10, "estimator_key": "pdis", "estimator_label": "PDIS", "estimator_family": "is_like", "ci_method": "bootstrap_percentile", "ci_level": 0.9, "estimate": 1.8, "true_value": 0.8, "error": 1.0, "abs_error": 1.0, "squared_error": 1.0, "shared_wess": 2.0, "wess_native_applicable": True, "ci_low": 1.7, "ci_high": 1.9, "ci_width": 0.2, "covered": 0.0, "sample_size": 100, "mismatch_alpha": 0.8, "seed": 0},
            {"experiment_id": "experiment_3", "experiment_title": "x", "env_name": "short_tabular_mdp", "condition_id": "c", "replicate_id": 0, "dataset_seed": 1, "estimator_key": "dr", "estimator_label": "DR", "estimator_family": "doubly_robust", "ci_method": "bootstrap_percentile", "ci_level": 0.9, "estimate": 0.82, "true_value": 0.8, "error": 0.02, "abs_error": 0.02, "squared_error": 0.0004, "shared_wess": 12.0, "wess_native_applicable": False, "ci_low": 0.75, "ci_high": 0.89, "ci_width": 0.14, "covered": 1.0, "sample_size": 300, "mismatch_alpha": 0.4, "seed": 0},
            {"experiment_id": "experiment_3", "experiment_title": "x", "env_name": "short_tabular_mdp", "condition_id": "c", "replicate_id": 1, "dataset_seed": 2, "estimator_key": "dr", "estimator_label": "DR", "estimator_family": "doubly_robust", "ci_method": "bootstrap_percentile", "ci_level": 0.9, "estimate": 0.78, "true_value": 0.8, "error": -0.02, "abs_error": 0.02, "squared_error": 0.0004, "shared_wess": 10.0, "wess_native_applicable": False, "ci_low": 0.71, "ci_high": 0.85, "ci_width": 0.14, "covered": 1.0, "sample_size": 300, "mismatch_alpha": 0.4, "seed": 0},
        ]
    )
    artifacts = generate_study_artifacts(raw, output_dir=tmp_path, primary_level=0.9)
    assert not artifacts["condition_summary"].empty
    assert {
        "spearman_wess_abs_error",
        "spearman_width_abs_error",
        "mean_abs_spearman_wess_abs_error",
        "mean_abs_spearman_width_abs_error",
        "share_expected_sign_wess",
        "share_expected_sign_width",
    } <= set(artifacts["table_2_diagnostic_usefulness"].columns)
    pdis_row = artifacts["table_2_diagnostic_usefulness"].set_index("estimator_key").loc["pdis"]
    assert np.isclose(pdis_row["spearman_wess_abs_error"], 1.0)
    assert np.isclose(pdis_row["mean_abs_spearman_wess_abs_error"], 1.0)
    assert (tmp_path / "table_1_main_summary.csv").exists()
    assert (tmp_path / "table_2_diagnostic_usefulness.csv").exists()
