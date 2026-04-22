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

    order_variance = ["low", "medium", "high"]
    order_sample = sorted(point_df["sample_size"].dropna().unique()) if "sample_size" in point_df.columns else []
    order_mismatch = ["low", "medium", "high"]

    exp1 = raw_df[raw_df["experiment_id"] == "experiment_1"].copy()
    if not exp1.empty:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        plot_df = exp1[(exp1["estimator_key"].isin(["is", "snis"])) & (exp1["ci_method"] == "point")].copy()
        if "sample_size" in plot_df.columns and (plot_df["sample_size"] == 1000).any():
            plot_df = plot_df[plot_df["sample_size"] == 1000]
        if "mismatch_level" in plot_df.columns and (plot_df["mismatch_level"] == "medium").any():
            plot_df = plot_df[plot_df["mismatch_level"] == "medium"]
        wess = plot_df.groupby(["reward_variance_regime", "estimator_key"], dropna=False)["shared_wess"].mean().reset_index()
        rmse = plot_df.groupby(["reward_variance_regime", "estimator_key"], dropna=False)["squared_error"].mean().reset_index()
        rmse["rmse"] = np.sqrt(rmse["squared_error"])
        width = exp1[
            (exp1["estimator_key"].isin(["is", "snis"]))
            & (exp1["ci_method"] == "bootstrap_percentile")
            & (np.isclose(exp1["ci_level"], float(primary_level), equal_nan=False))
        ].copy()
        if "sample_size" in width.columns and (width["sample_size"] == 1000).any():
            width = width[width["sample_size"] == 1000]
        if "mismatch_level" in width.columns and (width["mismatch_level"] == "medium").any():
            width = width[width["mismatch_level"] == "medium"]
        width = width.groupby(["reward_variance_regime", "estimator_key"], dropna=False)["ci_width"].mean().reset_index()
        for estimator_key, sub in wess.groupby("estimator_key", dropna=False):
            sub = sub.set_index("reward_variance_regime").reindex(order_variance).reset_index()
            axes[0].plot(sub["reward_variance_regime"], sub["shared_wess"], marker="o", label=DISPLAY_NAMES[estimator_key])
        for estimator_key, sub in rmse.groupby("estimator_key", dropna=False):
            sub = sub.set_index("reward_variance_regime").reindex(order_variance).reset_index()
            axes[1].plot(sub["reward_variance_regime"], sub["rmse"], marker="o", label=DISPLAY_NAMES[estimator_key])
        for estimator_key, sub in width.groupby("estimator_key", dropna=False):
            sub = sub.set_index("reward_variance_regime").reindex(order_variance).reset_index()
            axes[2].plot(sub["reward_variance_regime"], sub["ci_width"], marker="o", label=DISPLAY_NAMES[estimator_key])
        axes[0].set_title("Figure 1A: Mean WESS")
        axes[1].set_title("Figure 1B: RMSE")
        axes[2].set_title("Figure 1C: Mean CI Width")
        for ax in axes:
            ax.set_xlabel("Reward Variance Regime")
        axes[0].set_ylabel("Mean WESS")
        axes[1].set_ylabel("RMSE")
        axes[2].set_ylabel("Mean CI Width")
        axes[2].legend(frameon=True, fontsize=8)
        fig.tight_layout()
        save_figure(fig, output_dir, "figure_1")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    plotted_any = False
    for ax, estimator_key in zip(axes, ["is", "snis", "pdis"]):
        sub = point_df[point_df["estimator_key"] == estimator_key].copy()
        if sub.empty:
            continue
        plotted_any = True
        ax.scatter(sub["shared_wess"], sub["abs_error"], s=12, alpha=0.35)
        _binned_trend(ax, sub["shared_wess"].to_numpy(), sub["abs_error"].to_numpy())
        corr = sub["shared_wess"].corr(sub["abs_error"], method="spearman")
        ax.set_title(f"{DISPLAY_NAMES[estimator_key]} (rho={corr:.2f})")
        ax.set_xlabel("WESS")
        ax.set_ylabel("Absolute Error")
    if plotted_any:
        fig.tight_layout()
        save_figure(fig, output_dir, "figure_2")
    else:
        plt.close(fig)

    exp3 = point_df[point_df["experiment_id"] == "experiment_3"].copy()
    if not exp3.empty:
        if "sample_size" in exp3.columns and (exp3["sample_size"] == 300).any():
            exp3 = exp3[exp3["sample_size"] == 300]
        fig, ax = plt.subplots(figsize=(7.5, 5.0))
        for estimator_key, sub in exp3.groupby("estimator_key", dropna=False):
            ax.scatter(sub["shared_wess"], sub["abs_error"], s=18, alpha=0.5, label=DISPLAY_NAMES[estimator_key])
        ax.set_xlabel("WESS")
        ax.set_ylabel("Absolute Error")
        ax.set_title("Figure 3: Cross-Family WESS vs Absolute Error")
        ax.legend(frameon=True, fontsize=8, ncol=2)
        fig.tight_layout()
        save_figure(fig, output_dir, "figure_3")

    if not table_2.empty:
        heat = table_2.set_index("estimator_key")[
            [
                "spearman_wess_abs_error",
                "spearman_wess_squared_error",
                "spearman_width_abs_error",
                "spearman_width_squared_error",
            ]
        ]
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        im = ax.imshow(heat.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-1.0, vmax=1.0)
        ax.set_xticks(np.arange(len(heat.columns)), labels=["WESS/AE", "WESS/SE", "Width/AE", "Width/SE"])
        ax.set_yticks(np.arange(len(heat.index)), labels=[DISPLAY_NAMES.get(x, x.upper()) for x in heat.index])
        ax.set_title("Figure 4: Diagnostic Correlation Heatmap")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        save_figure(fig, output_dir, "figure_4")

    exp3_interval = raw_df[
        (raw_df["experiment_id"] == "experiment_3")
        & (raw_df["ci_method"] == "bootstrap_percentile")
        & (np.isclose(raw_df["ci_level"], float(primary_level), equal_nan=False))
    ].copy()
    if not exp3_interval.empty:
        cover = exp3_interval.groupby(["sample_size", "estimator_key"], dropna=False)["covered"].mean().reset_index()
        width = exp3_interval.groupby(["sample_size", "estimator_key"], dropna=False)["ci_width"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(7.5, 5.0))
        for estimator_key, sub in cover.groupby("estimator_key", dropna=False):
            sub = sub.set_index("sample_size").reindex(order_sample).reset_index()
            ax.plot(sub["sample_size"], sub["covered"], marker="o", label=DISPLAY_NAMES[estimator_key])
        ax.axhline(float(primary_level), color="black", linestyle="--", linewidth=1.0)
        ax.set_xlabel("Episodes")
        ax.set_ylabel("Empirical Coverage")
        ax.set_title("Figure 5: Coverage vs Sample Size")
        ax.legend(frameon=True, fontsize=8, ncol=2)
        fig.tight_layout()
        save_figure(fig, output_dir, "figure_5")

        fig, ax = plt.subplots(figsize=(7.5, 5.0))
        for estimator_key, sub in width.groupby("estimator_key", dropna=False):
            sub = sub.set_index("sample_size").reindex(order_sample).reset_index()
            ax.plot(sub["sample_size"], sub["ci_width"], marker="o", label=DISPLAY_NAMES[estimator_key])
        ax.set_xlabel("Episodes")
        ax.set_ylabel("Mean CI Width")
        ax.set_title("Figure 6: Interval Width vs Sample Size")
        ax.legend(frameon=True, fontsize=8, ncol=2)
        fig.tight_layout()
        save_figure(fig, output_dir, "figure_6")

    exp4_points = point_df[point_df["experiment_id"] == "experiment_4"].copy()
    exp4_interval = raw_df[
        (raw_df["experiment_id"] == "experiment_4")
        & (raw_df["ci_method"] == "bootstrap_percentile")
        & (np.isclose(raw_df["ci_level"], float(primary_level), equal_nan=False))
    ].copy()
    if not exp4_points.empty:
        rmse = exp4_points.groupby(["mismatch_level", "estimator_key"], dropna=False)["squared_error"].mean().reset_index()
        rmse["rmse"] = np.sqrt(rmse["squared_error"])
        wess = exp4_points.groupby(["mismatch_level", "estimator_key"], dropna=False)["shared_wess"].mean().reset_index()
        width = exp4_interval.groupby(["mismatch_level", "estimator_key"], dropna=False)["ci_width"].mean().reset_index()
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        for estimator_key, sub in rmse.groupby("estimator_key", dropna=False):
            sub = sub.set_index("mismatch_level").reindex(order_mismatch).reset_index()
            axes[0].plot(sub["mismatch_level"], sub["rmse"], marker="o", label=DISPLAY_NAMES[estimator_key])
        for estimator_key, sub in wess.groupby("estimator_key", dropna=False):
            sub = sub.set_index("mismatch_level").reindex(order_mismatch).reset_index()
            axes[1].plot(sub["mismatch_level"], sub["shared_wess"], marker="o", label=DISPLAY_NAMES[estimator_key])
        for estimator_key, sub in width.groupby("estimator_key", dropna=False):
            sub = sub.set_index("mismatch_level").reindex(order_mismatch).reset_index()
            axes[2].plot(sub["mismatch_level"], sub["ci_width"], marker="o", label=DISPLAY_NAMES[estimator_key])
        axes[0].set_title("Figure 7A: RMSE")
        axes[1].set_title("Figure 7B: WESS")
        axes[2].set_title("Figure 7C: Mean CI Width")
        axes[0].set_ylabel("RMSE")
        axes[1].set_ylabel("WESS")
        axes[2].set_ylabel("Mean CI Width")
        for ax in axes:
            ax.set_xlabel("Mismatch Level")
        axes[2].legend(frameon=True, fontsize=8, ncol=2)
        fig.tight_layout()
        save_figure(fig, output_dir, "figure_7")

    exp5 = raw_df[
        (raw_df["experiment_id"] == "experiment_5")
        & (raw_df["estimator_key"] == "fqe")
        & (raw_df["ci_method"] == "bootstrap_percentile")
        & (np.isclose(raw_df["ci_level"], float(primary_level), equal_nan=False))
    ].copy()
    if not exp5.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        cov = exp5.groupby(["calibration_env", "sample_size"], dropna=False)["covered"].mean().reset_index()
        wid = exp5.groupby(["calibration_env", "sample_size"], dropna=False)["ci_width"].mean().reset_index()
        for env_name, sub in cov.groupby("calibration_env", dropna=False):
            axes[0].plot(sub["sample_size"], sub["covered"], marker="o", label=str(env_name))
        for env_name, sub in wid.groupby("calibration_env", dropna=False):
            axes[1].plot(sub["sample_size"], sub["ci_width"], marker="o", label=str(env_name))
        axes[0].axhline(float(primary_level), color="black", linestyle="--", linewidth=1.0)
        axes[0].set_title("Figure 8A: FQE Coverage")
        axes[1].set_title("Figure 8B: FQE Width")
        axes[0].set_xlabel("Episodes")
        axes[1].set_xlabel("Episodes")
        axes[0].set_ylabel("Empirical Coverage")
        axes[1].set_ylabel("Mean CI Width")
        axes[1].legend(frameon=True, fontsize=8)
        fig.tight_layout()
        save_figure(fig, output_dir, "figure_8")
