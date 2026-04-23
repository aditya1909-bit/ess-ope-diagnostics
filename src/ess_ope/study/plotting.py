from __future__ import annotations

from pathlib import Path

from ess_ope.plotting._backend import ensure_headless_backend

ensure_headless_backend()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ess_ope.plotting.utils import save_figure
from ess_ope.study.estimators import DISPLAY_NAMES


def _style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#f6f2ea",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
        }
    )


def _binned_trend(ax: plt.Axes, x: np.ndarray, y: np.ndarray, bins: int = 10) -> None:
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 5:
        return
    x = x[mask]
    y = y[mask]
    edges = np.quantile(x, np.linspace(0.0, 1.0, min(bins, len(np.unique(x))) + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return
    centers = []
    means = []
    for left, right in zip(edges[:-1], edges[1:]):
        in_bin = (x >= left) & (x <= right if right == edges[-1] else x < right)
        if np.any(in_bin):
            centers.append(float(np.mean(x[in_bin])))
            means.append(float(np.mean(y[in_bin])))
    if centers:
        ax.plot(centers, means, color="black", linewidth=2.0)


def _mean_se(df: pd.DataFrame, value_col: str, group_cols: list[str]) -> pd.DataFrame:
    grouped = df.groupby(group_cols, dropna=False)[value_col]
    out = grouped.agg(["mean", "std", "count"]).reset_index()
    out["se"] = out["std"].fillna(0.0) / np.sqrt(np.maximum(out["count"], 1))
    return out


def generate_study_figures(
    raw_df: pd.DataFrame,
    point_df: pd.DataFrame,
    table_2: pd.DataFrame,
    output_dir: str | Path,
    primary_level: float,
) -> None:
    _style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exp1_points = raw_df[(raw_df["experiment_id"] == "experiment_1") & (raw_df["ci_method"] == "point")].copy()
    exp1_interval = raw_df[
        (raw_df["experiment_id"] == "experiment_1")
        & (raw_df["ci_method"] == "bootstrap_percentile")
        & (np.isclose(raw_df["ci_level"], float(primary_level), equal_nan=False))
    ].copy()
    if not exp1_points.empty:
        exp1_points = exp1_points[exp1_points["estimator_key"].isin(["is", "snis"])].copy()
        exp1_interval = exp1_interval[exp1_interval["estimator_key"].isin(["is", "snis"])].copy()
        if (exp1_points["sample_size"] == 500).any():
            exp1_points = exp1_points[exp1_points["sample_size"] == 500]
            exp1_interval = exp1_interval[exp1_interval["sample_size"] == 500]
        fig, axes = plt.subplots(1, 4, figsize=(17, 4.5))
        x_col = "reward_variance_scale"
        wess = exp1_points.groupby([x_col, "estimator_key"], dropna=False)["shared_wess"].mean().reset_index()
        est_var = exp1_points.groupby([x_col, "estimator_key"], dropna=False)["estimate"].var(ddof=0).reset_index(name="estimator_variance")
        abs_error = exp1_points.groupby([x_col, "estimator_key"], dropna=False)["abs_error"].mean().reset_index()
        width = exp1_interval.groupby([x_col, "estimator_key"], dropna=False)["ci_width"].mean().reset_index()
        for estimator_key, sub in wess.groupby("estimator_key", dropna=False):
            sub = sub.sort_values(x_col)
            axes[0].plot(sub[x_col], sub["shared_wess"], marker="o", label=DISPLAY_NAMES[estimator_key])
        for estimator_key, sub in est_var.groupby("estimator_key", dropna=False):
            sub = sub.sort_values(x_col)
            axes[1].plot(sub[x_col], sub["estimator_variance"], marker="o", label=DISPLAY_NAMES[estimator_key])
        for estimator_key, sub in abs_error.groupby("estimator_key", dropna=False):
            sub = sub.sort_values(x_col)
            axes[2].plot(sub[x_col], sub["abs_error"], marker="o", label=DISPLAY_NAMES[estimator_key])
        for estimator_key, sub in width.groupby("estimator_key", dropna=False):
            sub = sub.sort_values(x_col)
            axes[3].plot(sub[x_col], sub["ci_width"], marker="o", label=DISPLAY_NAMES[estimator_key])
        axes[0].set_title("Figure 1A: Mean WESS")
        axes[1].set_title("Figure 1B: Estimator Variance")
        axes[2].set_title("Figure 1C: Mean Absolute Error")
        axes[3].set_title("Figure 1D: Mean CI Width")
        axes[0].set_ylabel("Mean WESS")
        axes[1].set_ylabel("Variance")
        axes[2].set_ylabel("Absolute Error")
        axes[3].set_ylabel("CI Width")
        for ax in axes:
            ax.set_xlabel("Reward Variance Scale")
        axes[3].legend(frameon=True, fontsize=8)
        fig.tight_layout()
        save_figure(fig, output_dir, "figure_1")

    exp2 = point_df[point_df["experiment_id"] == "experiment_2"].copy()
    if not exp2.empty:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for ax, estimator_key in zip(axes, ["is", "snis"]):
            sub = exp2[exp2["estimator_key"] == estimator_key].copy()
            if sub.empty:
                continue
            ax.scatter(sub["shared_wess"], sub["abs_error"], s=14, alpha=0.35)
            _binned_trend(ax, sub["shared_wess"].to_numpy(), sub["abs_error"].to_numpy())
            corr = sub["shared_wess"].corr(sub["abs_error"], method="spearman")
            ax.set_title(f"{DISPLAY_NAMES[estimator_key]} (rho={corr:.2f})")
            ax.set_xlabel("WESS")
            ax.set_ylabel("Absolute Error")
        fig.tight_layout()
        save_figure(fig, output_dir, "figure_2")

    if not table_2.empty:
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        table_2 = table_2.reset_index(drop=True)
        x = np.arange(len(table_2))
        width = 0.35
        ax.bar(
            x - width / 2,
            table_2["mean_abs_spearman_wess_abs_error"],
            width=width,
            yerr=table_2["se_abs_spearman_wess_abs_error"],
            capsize=3,
            label="Mean |Spearman(WESS, AbsErr)|",
        )
        ax.bar(
            x + width / 2,
            table_2["mean_abs_spearman_width_abs_error"],
            width=width,
            yerr=table_2["se_abs_spearman_width_abs_error"],
            capsize=3,
            label="Mean |Spearman(CI Width, AbsErr)|",
        )
        ax.set_xticks(x, labels=[DISPLAY_NAMES.get(key, key.upper()) for key in table_2["estimator_key"]])
        ax.set_ylabel("Condition-Matched Correlation Strength")
        ax.set_title("Figure 3: Cross-Family Diagnostic Strength Summary")
        ax.legend(frameon=True, fontsize=9)
        fig.tight_layout()
        save_figure(fig, output_dir, "figure_3")

    exp3 = point_df[point_df["experiment_id"] == "experiment_3"].copy()
    if not exp3.empty:
        appendix = exp3.copy()
        if (appendix["sample_size"] == 300).any():
            appendix = appendix[appendix["sample_size"] == 300]
        fig, ax = plt.subplots(figsize=(7.5, 5.0))
        for estimator_key, sub in appendix.groupby("estimator_key", dropna=False):
            ax.scatter(sub["shared_wess"], sub["abs_error"], s=18, alpha=0.45, label=DISPLAY_NAMES[estimator_key])
        ax.set_xlabel("WESS")
        ax.set_ylabel("Absolute Error")
        ax.set_title("Appendix Figure A1: Cross-Family WESS vs Absolute Error")
        ax.legend(frameon=True, fontsize=8, ncol=2)
        fig.tight_layout()
        save_figure(fig, output_dir, "appendix_figure_a1")

    exp4_points = point_df[point_df["experiment_id"] == "experiment_4"].copy()
    exp4_interval = raw_df[
        (raw_df["experiment_id"] == "experiment_4")
        & (raw_df["ci_method"] == "bootstrap_percentile")
        & (np.isclose(raw_df["ci_level"], float(primary_level), equal_nan=False))
    ].copy()
    if not exp4_points.empty:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        err = exp4_points.groupby(["mismatch_alpha", "estimator_key"], dropna=False)["abs_error"].mean().reset_index()
        wess = exp4_points.groupby(["mismatch_alpha", "estimator_key"], dropna=False)["shared_wess"].mean().reset_index()
        width = exp4_interval.groupby(["mismatch_alpha", "estimator_key"], dropna=False)["ci_width"].mean().reset_index()
        for estimator_key, sub in err.groupby("estimator_key", dropna=False):
            sub = sub.sort_values("mismatch_alpha")
            axes[0].plot(sub["mismatch_alpha"], sub["abs_error"], marker="o", label=DISPLAY_NAMES[estimator_key])
        for estimator_key, sub in wess.groupby("estimator_key", dropna=False):
            sub = sub.sort_values("mismatch_alpha")
            axes[1].plot(sub["mismatch_alpha"], sub["shared_wess"], marker="o", label=DISPLAY_NAMES[estimator_key])
        for estimator_key, sub in width.groupby("estimator_key", dropna=False):
            sub = sub.sort_values("mismatch_alpha")
            axes[2].plot(sub["mismatch_alpha"], sub["ci_width"], marker="o", label=DISPLAY_NAMES[estimator_key])
        axes[0].set_title("Figure 4A: Mean Absolute Error")
        axes[1].set_title("Figure 4B: Mean WESS")
        axes[2].set_title("Figure 4C: Mean CI Width")
        axes[0].set_ylabel("Absolute Error")
        axes[1].set_ylabel("WESS")
        axes[2].set_ylabel("CI Width")
        for ax in axes:
            ax.set_xlabel("Mismatch Alpha")
        axes[2].legend(frameon=True, fontsize=8, ncol=2)
        fig.tight_layout()
        save_figure(fig, output_dir, "figure_4")

    exp3_interval = raw_df[
        (raw_df["experiment_id"] == "experiment_3")
        & (raw_df["ci_method"] == "bootstrap_percentile")
        & (np.isclose(raw_df["ci_level"], float(primary_level), equal_nan=False))
    ].copy()
    if not exp3_interval.empty:
        cover = _mean_se(exp3_interval, "covered", ["sample_size", "estimator_key"])
        width = _mean_se(exp3_interval, "ci_width", ["sample_size", "estimator_key"])
        fig, ax = plt.subplots(figsize=(7.5, 5.0))
        for estimator_key, sub in cover.groupby("estimator_key", dropna=False):
            sub = sub.sort_values("sample_size")
            ax.plot(sub["sample_size"], sub["mean"], marker="o", label=DISPLAY_NAMES[estimator_key])
        ax.axhline(float(primary_level), color="black", linestyle="--", linewidth=1.0)
        ax.set_xlabel("Episodes")
        ax.set_ylabel("Empirical Coverage")
        ax.set_title("Figure 5: Empirical 90% Coverage vs Sample Size")
        ax.legend(frameon=True, fontsize=8, ncol=2)
        fig.tight_layout()
        save_figure(fig, output_dir, "figure_5")

        fig, ax = plt.subplots(figsize=(7.5, 5.0))
        for estimator_key, sub in width.groupby("estimator_key", dropna=False):
            sub = sub.sort_values("sample_size")
            ax.errorbar(sub["sample_size"], sub["mean"], yerr=sub["se"], marker="o", capsize=3, label=DISPLAY_NAMES[estimator_key])
        ax.set_xlabel("Episodes")
        ax.set_ylabel("Mean CI Width")
        ax.set_title("Figure 6: Mean 90% CI Width vs Sample Size")
        ax.legend(frameon=True, fontsize=8, ncol=2)
        fig.tight_layout()
        save_figure(fig, output_dir, "figure_6")

    exp5 = raw_df[
        (raw_df["experiment_id"] == "experiment_5")
        & (raw_df["estimator_key"] == "fqe")
        & (raw_df["ci_method"] == "bootstrap_percentile")
        & (np.isclose(raw_df["ci_level"], float(primary_level), equal_nan=False))
    ].copy()
    if not exp5.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        cov = exp5.groupby(["calibration_env", "sample_size"], dropna=False)["covered"].mean().reset_index()
        wid = _mean_se(exp5, "ci_width", ["calibration_env", "sample_size"])
        for env_name, sub in cov.groupby("calibration_env", dropna=False):
            sub = sub.sort_values("sample_size")
            axes[0].plot(sub["sample_size"], sub["covered"], marker="o", label=str(env_name))
        for env_name, sub in wid.groupby("calibration_env", dropna=False):
            sub = sub.sort_values("sample_size")
            axes[1].errorbar(sub["sample_size"], sub["mean"], yerr=sub["se"], marker="o", capsize=3, label=str(env_name))
        axes[0].axhline(float(primary_level), color="black", linestyle="--", linewidth=1.0)
        axes[0].set_title("Figure 7A: FQE Coverage")
        axes[1].set_title("Figure 7B: FQE Width")
        axes[0].set_xlabel("Episodes")
        axes[1].set_xlabel("Episodes")
        axes[0].set_ylabel("Empirical Coverage")
        axes[1].set_ylabel("Mean CI Width")
        axes[1].legend(frameon=True, fontsize=9)
        fig.tight_layout()
        save_figure(fig, output_dir, "figure_7")
