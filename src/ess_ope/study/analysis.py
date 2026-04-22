from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from ess_ope.utils.logging import save_results_table


PRIMARY_INTERVAL_METHOD = "bootstrap_percentile"
MAIN_EXPERIMENT_ID = "experiment_3"
MAIN_SAMPLE_SIZE = 300


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
        "mismatch_level",
        "reward_variance_regime",
        "reward_mean_structure",
        "reward_variant",
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
    if width_proxy.empty:
        width_proxy = raw_df[raw_df["ci_method"] != "point"][
            ["condition_id", "replicate_id", "estimator_key", "ci_width", "covered"]
        ].rename(columns={"ci_width": "ci_width_proxy", "covered": "coverage_proxy"})
    width_proxy = width_proxy.drop_duplicates(subset=["condition_id", "replicate_id", "estimator_key"])
    point_df = point_df.merge(width_proxy, on=["condition_id", "replicate_id", "estimator_key"], how="left")
    point_df["coverage_proxy"] = point_df["coverage_proxy"].astype(float)
    return point_df


def build_condition_summary(raw_df: pd.DataFrame) -> pd.DataFrame:
    point_df = raw_df[raw_df["ci_method"] == "point"].copy()
    if point_df.empty:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    interval_df = raw_df[raw_df["ci_method"] != "point"].copy()
    group_cols = _condition_cols(point_df) + ["estimator_key"]
    for key, group in point_df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        row["bias"] = float(group["error"].mean())
        row["variance"] = float(group["estimate"].var(ddof=0))
        row["rmse"] = float(np.sqrt(group["squared_error"].mean()))
        row["mean_wess"] = float(group["shared_wess"].mean())
        for ci_level in sorted(interval_df["ci_level"].dropna().unique()):
            level_mask = np.isclose(interval_df["ci_level"], float(ci_level))
            sub = interval_df[level_mask].copy()
            for col in group_cols:
                sub = sub[sub[col] == row[col]]
            row[f"mean_ci_width_{ci_level:.2f}"] = float(sub["ci_width"].mean()) if not sub.empty else np.nan
            row[f"coverage_{ci_level:.2f}"] = float(sub["covered"].mean()) if not sub.empty else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def _main_experiment_point_df(raw_df: pd.DataFrame, primary_level: float) -> pd.DataFrame:
    point_df = prepare_point_table(raw_df, primary_level=primary_level)
    if point_df.empty:
        return point_df
    sub = point_df[point_df["experiment_id"] == MAIN_EXPERIMENT_ID].copy()
    if "sample_size" in sub.columns and (sub["sample_size"] == MAIN_SAMPLE_SIZE).any():
        sub = sub[sub["sample_size"] == MAIN_SAMPLE_SIZE].copy()
    return sub


def _main_interval_df(raw_df: pd.DataFrame, ci_level: float) -> pd.DataFrame:
    interval_df = raw_df[
        (raw_df["experiment_id"] == MAIN_EXPERIMENT_ID)
        & (raw_df["ci_method"] != "point")
        & (np.isclose(raw_df["ci_level"], float(ci_level), equal_nan=False))
    ].copy()
    if "sample_size" in interval_df.columns and (interval_df["sample_size"] == MAIN_SAMPLE_SIZE).any():
        interval_df = interval_df[interval_df["sample_size"] == MAIN_SAMPLE_SIZE].copy()
    return interval_df


def build_table_1_main_summary(raw_df: pd.DataFrame, primary_level: float) -> pd.DataFrame:
    point_df = _main_experiment_point_df(raw_df, primary_level)
    if point_df.empty:
        return pd.DataFrame()
    rows = []
    interval_df = _main_interval_df(raw_df, primary_level)
    interval_95 = _main_interval_df(raw_df, 0.95)
    for estimator_key, group in point_df.groupby("estimator_key", dropna=False):
        sub_interval = interval_df[interval_df["estimator_key"] == estimator_key]
        sub_95 = interval_95[interval_95["estimator_key"] == estimator_key]
        row = {
            "estimator_key": estimator_key,
            "bias": float(group["error"].mean()),
            "variance": float(group["estimate"].var(ddof=0)),
            "rmse": float(np.sqrt(group["squared_error"].mean())),
            "mean_wess": float(group["shared_wess"].mean()),
            "mean_ci_width": float(sub_interval["ci_width"].mean()) if not sub_interval.empty else np.nan,
            "coverage_0.90": float(sub_interval["covered"].mean()) if not sub_interval.empty else np.nan,
            "coverage_0.95": float(sub_95["covered"].mean()) if not sub_95.empty else np.nan,
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values("estimator_key").reset_index(drop=True)


def build_table_2_diagnostic_usefulness(raw_df: pd.DataFrame, primary_level: float) -> pd.DataFrame:
    point_df = _main_experiment_point_df(raw_df, primary_level)
    if point_df.empty:
        return pd.DataFrame()
    rows = []
    for estimator_key, group in point_df.groupby("estimator_key", dropna=False):
        rows.append(
            {
                "estimator_key": estimator_key,
                "spearman_wess_abs_error": _safe_spearman(group["shared_wess"], group["abs_error"]),
                "spearman_wess_squared_error": _safe_spearman(group["shared_wess"], group["squared_error"]),
                "spearman_width_abs_error": _safe_spearman(group["ci_width_proxy"], group["abs_error"]),
                "spearman_width_squared_error": _safe_spearman(group["ci_width_proxy"], group["squared_error"]),
            }
        )
    return pd.DataFrame(rows).sort_values("estimator_key").reset_index(drop=True)


def generate_study_artifacts(raw_df: pd.DataFrame, output_dir: str | Path, primary_level: float) -> Dict[str, pd.DataFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    point_df = prepare_point_table(raw_df, primary_level=primary_level)
    condition_summary = build_condition_summary(raw_df)
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
