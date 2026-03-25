from __future__ import annotations

import pandas as pd

from ess_ope.evaluation.summary import build_chain_bandit_sensitivity_summary


def test_chain_bandit_sensitivity_summary_captures_axes() -> None:
    df = pd.DataFrame(
        {
            "env_name": ["chain_bandit"] * 8,
            "alpha": [0.2] * 8,
            "beta": [0.1] * 8,
            "K": [100] * 8,
            "chain_variant": ["transitional"] * 8,
            "transition_strength": [0.5] * 8,
            "reward_mean_scale": [0.5, 0.5, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0],
            "reward_gap": [0.2] * 8,
            "reward_std": [0.1, 0.5, 0.1, 0.5, 0.1, 0.5, 0.1, 0.5],
            "ess_is": [40, 41, 39, 40, 42, 43, 41, 42],
            "ess_is_over_k": [0.40, 0.41, 0.39, 0.40, 0.42, 0.43, 0.41, 0.42],
            "estimate_is_pdis": [0.0] * 8,
            "error_is_pdis": [0.0] * 8,
            "abs_error_is_pdis": [0.10, 0.20, 0.08, 0.18, 0.11, 0.22, 0.09, 0.19],
            "estimate_dm_tabular": [0.0] * 8,
            "error_dm_tabular": [0.0] * 8,
            "abs_error_dm_tabular": [0.06, 0.12, 0.03, 0.10, 0.07, 0.13, 0.04, 0.11],
        }
    )

    summary = build_chain_bandit_sensitivity_summary(df)
    assert not summary.empty
    assert {"reward_mean_scale", "reward_std"}.issubset(set(summary["axis"]))
    assert {"is_pdis", "dm_tabular"}.issubset(set(summary["estimator"]))
