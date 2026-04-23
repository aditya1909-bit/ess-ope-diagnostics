from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from ess_ope.utils.logging import save_results_table


PRIMARY_INTERVAL_METHOD = "bootstrap_percentile"
MAIN_EXPERIMENT_ID = "experiment_3"
MAIN_SAMPLE_SIZE = 300
MAIN_MISMATCH_ALPHA = 0.4


def _safe_spearman(x: pd.Series, y: pd.Series) -> float:
    joined = pd.concat([x, y], axis=1).dropna()
    if len(joined) < 2:
        return np.nan
    if joined.iloc[:, 0].nunique() <= 1 or joined.iloc[:, 1].nunique() <= 1:
        return np.nan
    return float(joined.iloc[:, 0].corr(joined.iloc[:, 1], method="spearman"))


def _condition_cols(df: pd.DataFrame) -> List[str]:
    ordered = [
        "experiment_id",
        "env_name",
        "sample_size",
        "mismatch_alpha",
        "reward_variance_scale",
        "calibration_env",
        "seed",
    ]
    return [col for col in ordered if col in df.columns]


def prepare_point_table(raw_df: pd.DataFrame, primary_level: float) -> pd.DataFrame:
    point_df = raw_df[raw_df["ci_method"] == "point"].copy()
    if point_df.empty:
        return pd.DataFrame()
    width_proxy = raw_df[
        (raw_df["ci_method"] == PRIMARY_INTERVAL_METHOD)
        & (np.isclose(raw_df["ci_level"], float(primary_level), equal_nan=False))
    ][["condition_id", "replicate_id", "estimator_key", "ci_width", "covered"]].rename(
        columns={"ci_width": "ci_width_proxy", "covered": "coverage_proxy"}
    )
    width_proxy = width_proxy.drop_duplicates(subset=["condition_id", "replicate_id", "estimator_key"])
    point_df = point_df.merge(width_proxy, on=["condition_id", "replicate_id", "estimator_key"], how="left")
    return point_df


def build_condition_summary(raw_df: pd.DataFrame, primary_level: float) -> pd.DataFrame:
    point_df = prepare_point_table(raw_df, primary_level=primary_level)
    if point_df.empty:
        return pd.DataFrame()
    interval_df = raw_df[
        (raw_df["ci_method"] != "point")
        & (np.isclose(raw_df["ci_level"], float(primary_level), equal_nan=False))
    ].copy()
    rows: List[Dict[str, object]] = []
    group_cols = _condition_cols(point_df) + ["estimator_key", "estimator_label", "estimator_family"]
    for key, group in point_df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        sub_interval = interval_df.copy()
        for col in group_cols:
            sub_interval = sub_interval[sub_interval[col] == row[col]]
        row["mean_abs_error"] = float(group["abs_error"].mean())
        row["estimator_variance"] = float(group["estimate"].var(ddof=0))
        row["mean_wess"] = float(group["shared_wess"].mean())
        row["mean_ci_width"] = float(sub_interval["ci_width"].mean()) if not sub_interval.empty else np.nan
        row["empirical_coverage"] = float(sub_interval["covered"].mean()) if not sub_interval.empty else np.nan
        row["spearman_wess_abs_error"] = _safe_spearman(group["shared_wess"], group["abs_error"])
        row["spearman_width_abs_error"] = _safe_spearman(group["ci_width_proxy"], group["abs_error"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def _main_experiment_point_df(raw_df: pd.DataFrame, primary_level: float) -> pd.DataFrame:
    point_df = prepare_point_table(raw_df, primary_level=primary_level)
    if point_df.empty:
        return point_df
    sub = point_df[point_df["experiment_id"] == MAIN_EXPERIMENT_ID].copy()
    if "sample_size" in sub.columns and (sub["sample_size"] == MAIN_SAMPLE_SIZE).any():
        sub = sub[np.isclose(sub["sample_size"], MAIN_SAMPLE_SIZE)]
    if "mismatch_alpha" in sub.columns and np.isclose(sub["mismatch_alpha"], MAIN_MISMATCH_ALPHA).any():
        sub = sub[np.isclose(sub["mismatch_alpha"], MAIN_MISMATCH_ALPHA)]
    return sub


def _main_interval_df(raw_df: pd.DataFrame, ci_level: float) -> pd.DataFrame:
    interval_df = raw_df[
        (raw_df["experiment_id"] == MAIN_EXPERIMENT_ID)
        & (raw_df["ci_method"] != "point")
        & (np.isclose(raw_df["ci_level"], float(ci_level), equal_nan=False))
    ].copy()
    if "sample_size" in interval_df.columns and (interval_df["sample_size"] == MAIN_SAMPLE_SIZE).any():
        interval_df = interval_df[np.isclose(interval_df["sample_size"], MAIN_SAMPLE_SIZE)]
    if "mismatch_alpha" in interval_df.columns and np.isclose(interval_df["mismatch_alpha"], MAIN_MISMATCH_ALPHA).any():
        interval_df = interval_df[np.isclose(interval_df["mismatch_alpha"], MAIN_MISMATCH_ALPHA)]
    return interval_df


def build_table_1_main_summary(raw_df: pd.DataFrame, primary_level: float) -> pd.DataFrame:
    point_df = _main_experiment_point_df(raw_df, primary_level)
    if point_df.empty:
        return pd.DataFrame()
    interval_df = _main_interval_df(raw_df, primary_level)
    rows = []
    for estimator_key, group in point_df.groupby("estimator_key", dropna=False):
        sub_interval = interval_df[interval_df["estimator_key"] == estimator_key]
        rows.append(
            {
                "estimator_key": estimator_key,
                "estimator_label": str(group["estimator_label"].iloc[0]),
                "estimator_family": str(group["estimator_family"].iloc[0]),
                "mean_abs_error": float(group["abs_error"].mean()),
                "estimator_variance": float(group["estimate"].var(ddof=0)),
                "mean_wess": float(group["shared_wess"].mean()),
                "mean_ci_width": float(sub_interval["ci_width"].mean()) if not sub_interval.empty else np.nan,
                "empirical_coverage": float(sub_interval["covered"].mean()) if not sub_interval.empty else np.nan,
                "spearman_wess_abs_error": _safe_spearman(group["shared_wess"], group["abs_error"]),
                "spearman_width_abs_error": _safe_spearman(group["ci_width_proxy"], group["abs_error"]),
            }
        )
    return pd.DataFrame(rows).sort_values("estimator_key").reset_index(drop=True)


def build_table_2_diagnostic_usefulness(raw_df: pd.DataFrame, primary_level: float) -> pd.DataFrame:
    point_df = prepare_point_table(raw_df, primary_level=primary_level)
    if point_df.empty:
        return pd.DataFrame()
    sub = point_df[point_df["experiment_id"] == MAIN_EXPERIMENT_ID].copy()
    if sub.empty:
        return pd.DataFrame()

    condition_cols = [col for col in ["sample_size", "mismatch_alpha", "seed"] if col in sub.columns]
    per_condition_rows = []
    grouped = sub.groupby(condition_cols + ["estimator_key", "estimator_label", "estimator_family"], dropna=False)
    for key, group in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(condition_cols + ["estimator_key", "estimator_label", "estimator_family"], key))
        row["spearman_wess_abs_error"] = _safe_spearman(group["shared_wess"], group["abs_error"])
        row["spearman_width_abs_error"] = _safe_spearman(group["ci_width_proxy"], group["abs_error"])
        per_condition_rows.append(row)
    per_condition_df = pd.DataFrame(per_condition_rows)
    if per_condition_df.empty:
        return pd.DataFrame()

    rows = []
    for estimator_key, group in per_condition_df.groupby("estimator_key", dropna=False):
        wess = group["spearman_wess_abs_error"].dropna()
        width = group["spearman_width_abs_error"].dropna()
        rows.append(
            {
                "estimator_key": estimator_key,
                "estimator_label": str(group["estimator_label"].iloc[0]),
                "estimator_family": str(group["estimator_family"].iloc[0]),
                "spearman_wess_abs_error": float(wess.mean()) if not wess.empty else np.nan,
                "spearman_width_abs_error": float(width.mean()) if not width.empty else np.nan,
                "mean_abs_spearman_wess_abs_error": float(wess.abs().mean()) if not wess.empty else np.nan,
                "mean_abs_spearman_width_abs_error": float(width.abs().mean()) if not width.empty else np.nan,
                "se_abs_spearman_wess_abs_error": float(wess.abs().std(ddof=1) / np.sqrt(len(wess))) if len(wess) > 1 else 0.0,
                "se_abs_spearman_width_abs_error": float(width.abs().std(ddof=1) / np.sqrt(len(width))) if len(width) > 1 else 0.0,
                "share_expected_sign_wess": float((wess < 0).mean()) if not wess.empty else np.nan,
                "share_expected_sign_width": float((width > 0).mean()) if not width.empty else np.nan,
                "num_conditions_wess": int(len(wess)),
                "num_conditions_width": int(len(width)),
            }
        )
    return pd.DataFrame(rows).sort_values("estimator_key").reset_index(drop=True)


def generate_study_artifacts(raw_df: pd.DataFrame, output_dir: str | Path, primary_level: float) -> Dict[str, pd.DataFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    point_df = prepare_point_table(raw_df, primary_level=primary_level)
    condition_summary = build_condition_summary(raw_df, primary_level=primary_level)
    table_1 = build_table_1_main_summary(raw_df, primary_level=primary_level)
    table_2 = build_table_2_diagnostic_usefulness(raw_df, primary_level=primary_level)
    save_results_table(raw_df, output_dir, "replicate_results")
    save_results_table(point_df, output_dir, "point_estimates")
    save_results_table(condition_summary, output_dir, "condition_summary")
    save_results_table(table_1, output_dir, "table_1_main_summary")
    save_results_table(table_2, output_dir, "table_2_diagnostic_usefulness")
    return {
        "point_estimates": point_df,
        "condition_summary": condition_summary,
        "table_1_main_summary": table_1,
        "table_2_diagnostic_usefulness": table_2,
    }
