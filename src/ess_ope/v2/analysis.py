from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from ess_ope.utils.logging import save_results_table


CONDITION_COLS = [
    "phase_name",
    "env_family",
    "seed",
    "sample_size",
    "mismatch_level",
    "reward_noise_level",
    "support_regime",
    "horizon",
    "reward_regime",
    "rarity_level",
]


def _safe_corr(x: pd.Series, y: pd.Series, method: str) -> float:
    joined = pd.concat([x, y], axis=1).dropna()
    if len(joined) < 2:
        return np.nan
    if joined.iloc[:, 0].nunique() <= 1 or joined.iloc[:, 1].nunique() <= 1:
        return np.nan
    return float(joined.iloc[:, 0].corr(joined.iloc[:, 1], method=method))


def _rank(values: pd.Series, ascending: bool = True) -> pd.Series:
    return values.rank(method="average", ascending=ascending)


def auc_metrics(score: np.ndarray, label: np.ndarray) -> tuple[float, float]:
    score = np.asarray(score, dtype=float)
    label = np.asarray(label, dtype=float)
    mask = np.isfinite(score) & np.isfinite(label)
    score = score[mask]
    label = label[mask]
    if score.size == 0 or np.unique(label).size < 2:
        return np.nan, np.nan
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    pos = label > 0.5
    n_pos = float(np.sum(pos))
    n_neg = float(len(label) - np.sum(pos))
    if n_pos == 0 or n_neg == 0:
        return np.nan, np.nan
    auc = (np.sum(ranks[pos]) - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)

    desc = np.argsort(-score)
    sorted_label = label[desc]
    tp = np.cumsum(sorted_label)
    fp = np.cumsum(1.0 - sorted_label)
    precision = tp / np.maximum(1.0, tp + fp)
    recall = tp / n_pos
    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    auprc = float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))
    return float(auc), auprc


def build_estimator_summary(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame()
    point_df = prepare_point_table(raw_df)
    if point_df.empty:
        return pd.DataFrame()

    coverage_df = raw_df[raw_df["ci_method"] != "point"].copy()
    rows: List[Dict[str, object]] = []
    group_cols = [col for col in CONDITION_COLS if col in point_df.columns] + ["estimator_key"]
    for key, group in point_df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        errors = group["error"]
        sq = group["squared_error"]
        abs_err = group["abs_error"]
        row.update(
            {
                "bias": float(errors.mean()),
                "variance": float(group["estimate"].var(ddof=0)),
                "rmse": float(np.sqrt(sq.mean())),
                "mean_abs_error": float(abs_err.mean()),
                "mean_native_ess": float(group["native_diagnostic_value"].mean()) if group["native_diagnostic_kind"].notna().any() else np.nan,
                "diagnostic_rank_corr": _safe_corr(group["uncertainty_score"], abs_err, method="spearman"),
            }
        )
        for ci_level in sorted(coverage_df["ci_level"].dropna().unique()):
            sub = coverage_df[(coverage_df["ci_level"] == ci_level)]
            for col in group_cols:
                sub = sub[sub[col] == row[col]]
            row[f"avg_ci_width_{ci_level:.2f}"] = float(sub["ci_width"].mean()) if not sub.empty else np.nan
            row[f"coverage_{ci_level:.2f}"] = float(sub["covered"].mean()) if not sub.empty else np.nan
        rows.append(row)

    out = pd.DataFrame(rows)
    return out.sort_values(group_cols).reset_index(drop=True)


def build_condition_variance_summary(raw_df: pd.DataFrame) -> pd.DataFrame:
    point_df = raw_df[raw_df["ci_method"] == "point"].copy()
    if point_df.empty:
        return pd.DataFrame()
    group_cols = [col for col in CONDITION_COLS if col in point_df.columns] + ["estimator_key"]
    agg = (
        point_df.groupby(group_cols, dropna=False)
        .agg(
            empirical_variance=("estimate", lambda s: float(np.var(np.asarray(s, dtype=float), ddof=0))),
            mean_uncertainty_score=("uncertainty_score", "mean"),
            mean_ci_width=("ci_width_proxy", "mean"),
            mean_native_ess=("native_diagnostic_value", "mean"),
            mean_abs_error=("abs_error", "mean"),
            rmse=("squared_error", lambda s: float(np.sqrt(np.mean(np.asarray(s, dtype=float))))),
        )
        .reset_index()
    )
    return agg.sort_values(group_cols).reset_index(drop=True)


def build_diagnostic_quality_summary(raw_df: pd.DataFrame, condition_variance: pd.DataFrame) -> pd.DataFrame:
    point_df = raw_df[raw_df["ci_method"] == "point"].copy()
    if point_df.empty:
        return pd.DataFrame()

    rows = []
    for estimator_key, group in point_df.groupby("estimator_key", dropna=False):
        row: Dict[str, object] = {"estimator_key": estimator_key}
        has_ess = group["native_diagnostic_kind"].eq("ess").any()
        ess = group["native_diagnostic_value"]
        width = group["ci_width_proxy"]
        row["corr_ess_abs_error"] = _safe_corr(ess, group["abs_error"], "pearson") if has_ess else np.nan
        row["corr_ess_squared_error"] = _safe_corr(ess, group["squared_error"], "pearson") if has_ess else np.nan
        row["corr_ess_ci_width"] = _safe_corr(ess, width, "pearson") if has_ess else np.nan
        row["corr_ess_undercoverage"] = _safe_corr(ess, group["coverage_miss_proxy"], "pearson") if has_ess else np.nan
        row["corr_ci_width_abs_error"] = _safe_corr(width, group["abs_error"], "pearson")
        row["corr_ci_width_squared_error"] = _safe_corr(width, group["squared_error"], "pearson")
        row["corr_ci_width_undercoverage"] = _safe_corr(width, group["coverage_miss_proxy"], "pearson")

        var_sub = condition_variance[condition_variance["estimator_key"] == estimator_key]
        row["corr_ess_variance"] = _safe_corr(var_sub["mean_native_ess"], var_sub["empirical_variance"], "pearson") if has_ess else np.nan
        row["corr_ci_width_variance"] = _safe_corr(var_sub["mean_ci_width"], var_sub["empirical_variance"], "pearson")

        scope_to_labels = {
            "within_condition": ["label_sqerr_top10", "label_abs_error_top10", "label_ci_miss"],
            "global": ["label_sqerr_top10_global", "label_abs_error_top10_global", "label_ci_miss"],
        }
        for scope, label_cols in scope_to_labels.items():
            for label_col in label_cols:
                auroc, auprc = auc_metrics(width.to_numpy(), group[label_col].to_numpy())
                row[f"{scope}_width_{label_col}_auroc"] = auroc
                row[f"{scope}_width_{label_col}_auprc"] = auprc
                if has_ess:
                    auc_low, pr_low = auc_metrics((-ess).to_numpy(), group[label_col].to_numpy())
                    row[f"{scope}_low_ess_{label_col}_auroc"] = auc_low
                    row[f"{scope}_low_ess_{label_col}_auprc"] = pr_low
        rows.append(row)

    return pd.DataFrame(rows).sort_values("estimator_key").reset_index(drop=True)


def build_calibration_summary(raw_df: pd.DataFrame) -> pd.DataFrame:
    point_df = raw_df[raw_df["ci_method"] == "point"].copy()
    if point_df.empty:
        return pd.DataFrame()
    frames = []
    for estimator_key, group in point_df.groupby("estimator_key", dropna=False):
        if group["native_diagnostic_kind"].eq("ess").any() and group["native_diagnostic_value"].nunique(dropna=True) > 1:
            work = group.copy()
            work["bin_kind"] = "ess"
            work["bin_id"] = pd.qcut(work["native_diagnostic_value"], q=min(10, work["native_diagnostic_value"].nunique()), duplicates="drop")
            frames.append(work)
        if group["ci_width_proxy"].nunique(dropna=True) > 1:
            work = group.copy()
            work["bin_kind"] = "ci_width"
            work["bin_id"] = pd.qcut(work["ci_width_proxy"], q=min(10, work["ci_width_proxy"].nunique()), duplicates="drop")
            frames.append(work)
    if not frames:
        return pd.DataFrame()

    binned = pd.concat(frames, ignore_index=True)
    agg = (
        binned.groupby(["estimator_key", "bin_kind", "bin_id"], dropna=False)
        .agg(
            n=("estimate", "size"),
            mean_rmse=("squared_error", lambda s: float(np.sqrt(np.mean(np.asarray(s, dtype=float))))),
            mean_abs_error=("abs_error", "mean"),
            mean_squared_error=("squared_error", "mean"),
            empirical_coverage=("coverage_proxy", "mean"),
            mean_ci_width=("ci_width_proxy", "mean"),
            mean_native_ess=("native_diagnostic_value", "mean"),
        )
        .reset_index()
    )
    return agg.sort_values(["estimator_key", "bin_kind"]).reset_index(drop=True)


def build_failure_prediction_summary(raw_df: pd.DataFrame) -> pd.DataFrame:
    point_df = raw_df[raw_df["ci_method"] == "point"].copy()
    if point_df.empty:
        return pd.DataFrame()
    rows = []
    for estimator_key, group in point_df.groupby("estimator_key", dropna=False):
        has_ess = group["native_diagnostic_kind"].eq("ess").any()
        scope_to_labels = {
            "within_condition": ["label_sqerr_top10", "label_abs_error_top10", "label_ci_miss"],
            "global": ["label_sqerr_top10_global", "label_abs_error_top10_global", "label_ci_miss"],
        }
        for scope, label_names in scope_to_labels.items():
            for label_name in label_names:
                score_defs = [
                    ("wide_ci", group["ci_width_proxy"].to_numpy()),
                    ("narrow_ci", (-group["ci_width_proxy"]).to_numpy()),
                    ("uncertainty_score", group["uncertainty_score"].to_numpy()),
                ]
                if has_ess:
                    score_defs.append(("low_ess", (-group["native_diagnostic_value"]).to_numpy()))
                for score_name, score in score_defs:
                    auroc, auprc = auc_metrics(score, group[label_name].to_numpy())
                    rows.append(
                        {
                            "estimator_key": estimator_key,
                            "scope": scope,
                            "label_name": label_name,
                            "score_name": score_name,
                            "auroc": auroc,
                            "auprc": auprc,
                        }
                    )
    return pd.DataFrame(rows).sort_values(["estimator_key", "scope", "label_name", "score_name"]).reset_index(drop=True)


def build_cross_estimator_ranking(raw_df: pd.DataFrame) -> pd.DataFrame:
    point_df = raw_df[raw_df["ci_method"] == "point"].copy()
    if point_df.empty:
        return pd.DataFrame()
    group_cols = [col for col in CONDITION_COLS if col in point_df.columns]
    condition_summary = (
        point_df.groupby(group_cols + ["estimator_key"], dropna=False)
        .agg(rmse=("squared_error", lambda s: float(np.sqrt(np.mean(np.asarray(s, dtype=float))))), mean_abs_error=("abs_error", "mean"))
        .reset_index()
    )
    condition_summary["rank_rmse"] = condition_summary.groupby(group_cols)["rmse"].transform(lambda s: _rank(s, ascending=True))
    condition_summary["rank_abs_error"] = condition_summary.groupby(group_cols)["mean_abs_error"].transform(lambda s: _rank(s, ascending=True))
    agg = (
        condition_summary.groupby("estimator_key", dropna=False)
        .agg(avg_rank_rmse=("rank_rmse", "mean"), avg_rank_abs_error=("rank_abs_error", "mean"))
        .reset_index()
    )
    return agg.sort_values("avg_rank_rmse").reset_index(drop=True)


def build_diagnostic_correlation_table(raw_df: pd.DataFrame, condition_variance: pd.DataFrame) -> pd.DataFrame:
    point_df = raw_df[raw_df["ci_method"] == "point"].copy()
    rows = []
    for estimator_key, group in point_df.groupby("estimator_key", dropna=False):
        rows.extend(
            [
                {"estimator_key": estimator_key, "x_metric": "uncertainty_score", "y_metric": "abs_error", "correlation": _safe_corr(group["uncertainty_score"], group["abs_error"], "spearman")},
                {"estimator_key": estimator_key, "x_metric": "uncertainty_score", "y_metric": "squared_error", "correlation": _safe_corr(group["uncertainty_score"], group["squared_error"], "spearman")},
                {"estimator_key": estimator_key, "x_metric": "ci_width", "y_metric": "coverage_miss", "correlation": _safe_corr(group["ci_width_proxy"], group["coverage_miss_proxy"], "spearman")},
            ]
        )
        if group["native_diagnostic_kind"].eq("ess").any():
            rows.extend(
                [
                    {"estimator_key": estimator_key, "x_metric": "ess", "y_metric": "abs_error", "correlation": _safe_corr(group["native_diagnostic_value"], group["abs_error"], "spearman")},
                    {"estimator_key": estimator_key, "x_metric": "ess", "y_metric": "squared_error", "correlation": _safe_corr(group["native_diagnostic_value"], group["squared_error"], "spearman")},
                    {"estimator_key": estimator_key, "x_metric": "ess", "y_metric": "ci_width", "correlation": _safe_corr(group["native_diagnostic_value"], group["ci_width_proxy"], "spearman")},
                ]
            )
        var_sub = condition_variance[condition_variance["estimator_key"] == estimator_key]
        rows.append(
            {
                "estimator_key": estimator_key,
                "x_metric": "mean_ci_width",
                "y_metric": "empirical_variance",
                "correlation": _safe_corr(var_sub["mean_ci_width"], var_sub["empirical_variance"], "spearman"),
            }
        )
    return pd.DataFrame(rows)


def _add_failure_labels(point_df: pd.DataFrame) -> pd.DataFrame:
    work = point_df.copy()
    condition_cols = [col for col in CONDITION_COLS if col in work.columns] + ["estimator_key"]

    work["label_sqerr_top10"] = 0.0
    work["label_abs_error_top10"] = 0.0
    for _, idx in work.groupby(condition_cols).groups.items():
        sub = work.loc[idx]
        sq_thr = float(sub["squared_error"].quantile(0.9))
        abs_thr = float(sub["abs_error"].quantile(0.9))
        work.loc[idx, "label_sqerr_top10"] = (sub["squared_error"] >= sq_thr).astype(float)
        work.loc[idx, "label_abs_error_top10"] = (sub["abs_error"] >= abs_thr).astype(float)

    global_sq_thr = float(work["squared_error"].quantile(0.9))
    global_abs_thr = float(work["abs_error"].quantile(0.9))
    work["label_sqerr_top10_global"] = (work["squared_error"] >= global_sq_thr).astype(float)
    work["label_abs_error_top10_global"] = (work["abs_error"] >= global_abs_thr).astype(float)
    work["label_ci_miss"] = work["coverage_miss_proxy"].astype(float)
    return work


def prepare_point_table(raw_df: pd.DataFrame) -> pd.DataFrame:
    point_df = raw_df[raw_df["ci_method"] == "point"].copy()
    if point_df.empty:
        return pd.DataFrame()

    width_proxy = (
        raw_df[raw_df["ci_method"].eq("bootstrap_percentile")]
        .sort_values("ci_level")
        [["condition_id", "replicate_id", "estimator_key", "ci_width", "covered"]]
        .drop_duplicates(subset=["condition_id", "replicate_id", "estimator_key"], keep="last")
        .rename(columns={"ci_width": "ci_width_proxy", "covered": "coverage_proxy"})
    )
    if width_proxy.empty:
        width_proxy = (
            raw_df[raw_df["ci_method"] != "point"][["condition_id", "replicate_id", "estimator_key", "ci_width", "covered"]]
            .rename(columns={"ci_width": "ci_width_proxy", "covered": "coverage_proxy"})
            .drop_duplicates(subset=["condition_id", "replicate_id", "estimator_key"])
        )
    point_df = point_df.merge(width_proxy, on=["condition_id", "replicate_id", "estimator_key"], how="left")
    point_df["ci_width_proxy"] = point_df["ci_width_proxy"].fillna(np.nan)
    point_df["coverage_proxy"] = point_df["coverage_proxy"].fillna(np.nan)
    point_df["coverage_miss_proxy"] = 1.0 - point_df["coverage_proxy"]
    point_df["uncertainty_score"] = np.where(
        point_df["native_diagnostic_kind"].eq("ess"),
        -point_df["native_diagnostic_value"],
        point_df["ci_width_proxy"],
    )
    point_df = _add_failure_labels(point_df)
    return point_df


def generate_phase_artifacts(raw_df: pd.DataFrame, output_dir: str | Path) -> Dict[str, pd.DataFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    point_df = prepare_point_table(raw_df)
    if point_df.empty:
        artifacts = {
            "replicate_results": raw_df,
            "point_estimates": point_df,
            "table_a_estimator_summary": pd.DataFrame(),
            "table_b_diagnostic_quality": pd.DataFrame(),
            "calibration_summary": pd.DataFrame(),
            "failure_prediction_summary": pd.DataFrame(),
            "cross_estimator_ranking": pd.DataFrame(),
            "condition_variance_summary": pd.DataFrame(),
            "diagnostic_correlation_summary": pd.DataFrame(),
        }
        return artifacts

    condition_variance = build_condition_variance_summary(point_df)
    artifacts = {
        "replicate_results": raw_df,
        "point_estimates": point_df,
        "table_a_estimator_summary": build_estimator_summary(raw_df),
        "table_b_diagnostic_quality": build_diagnostic_quality_summary(point_df, condition_variance),
        "calibration_summary": build_calibration_summary(point_df),
        "failure_prediction_summary": build_failure_prediction_summary(point_df),
        "cross_estimator_ranking": build_cross_estimator_ranking(point_df),
        "condition_variance_summary": condition_variance,
        "diagnostic_correlation_summary": build_diagnostic_correlation_table(point_df, condition_variance),
    }
    for name, frame in artifacts.items():
        save_results_table(frame, output_dir, name)
    return artifacts
