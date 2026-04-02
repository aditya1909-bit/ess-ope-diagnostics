from __future__ import annotations

from pathlib import Path

from ess_ope.plotting._backend import ensure_headless_backend

ensure_headless_backend()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ess_ope.plotting.utils import save_figure


def _style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "figure.facecolor": "white",
            "axes.facecolor": "#f4f1eb",
        }
    )


def _scatter_panel(ax: plt.Axes, df: pd.DataFrame, x: str, y: str, title: str) -> None:
    if df.empty:
        return
    for estimator_key, sub in df.groupby("estimator_key", dropna=False):
        ax.scatter(sub[x], sub[y], s=12, alpha=0.45, label=estimator_key)
    ax.set_title(title)
    ax.set_xlabel(x.replace("_", " ").title())
    ax.set_ylabel(y.replace("_", " ").title())


def generate_v2_figures(
    point_df: pd.DataFrame,
    calibration_summary: pd.DataFrame,
    table_a: pd.DataFrame,
    table_b: pd.DataFrame,
    cross_ranking: pd.DataFrame,
    diagnostic_corr: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _style()

    if not point_df.empty:
        ess_df = point_df[point_df["native_diagnostic_kind"].eq("ess")].copy()
        if not ess_df.empty:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
            _scatter_panel(axes[0], ess_df, "native_diagnostic_value", "squared_error", "ESS vs Squared Error")
            _scatter_panel(axes[1], ess_df, "native_diagnostic_value", "ci_width_proxy", "ESS vs CI Width")
            axes[0].legend(frameon=True, fontsize=8)
            fig.tight_layout()
            save_figure(fig, output_dir, "v2_fig1_ess_scatter_panels")

        cal = calibration_summary.copy()
        if not cal.empty:
            for bin_kind, stem, ycol, title in [
                ("ess", "v2_fig2_ess_bins_vs_rmse", "mean_rmse", "ESS Bins vs RMSE"),
                ("ess", "v2_fig3_ess_bins_vs_coverage", "empirical_coverage", "ESS Bins vs Coverage"),
                ("ci_width", "v2_fig4_ci_width_bins_vs_coverage", "empirical_coverage", "CI-Width Bins vs Coverage"),
            ]:
                sub = cal[cal["bin_kind"] == bin_kind]
                if sub.empty:
                    continue
                fig, ax = plt.subplots(figsize=(7, 4.8))
                for estimator_key, grp in sub.groupby("estimator_key", dropna=False):
                    ax.plot(np.arange(len(grp)), grp[ycol], marker="o", label=estimator_key)
                ax.set_title(title)
                ax.set_xlabel("Bin")
                ax.set_ylabel(ycol.replace("_", " ").title())
                ax.legend(frameon=True, fontsize=8)
                fig.tight_layout()
                save_figure(fig, output_dir, stem)

        cov_cols = [col for col in table_a.columns if col.startswith("coverage_")]
        width_cols = [col for col in table_a.columns if col.startswith("avg_ci_width_")]
        if cov_cols:
            fig, ax = plt.subplots(figsize=(7, 4.8))
            nominal = [float(col.split("_")[-1]) for col in cov_cols]
            for estimator_key, grp in table_a.groupby("estimator_key", dropna=False):
                ax.plot(nominal, [grp[col].mean() for col in cov_cols], marker="o", label=estimator_key)
            ax.plot([min(nominal), max(nominal)], [min(nominal), max(nominal)], linestyle="--", color="black")
            ax.set_title("Nominal vs Empirical Coverage")
            ax.set_xlabel("Nominal")
            ax.set_ylabel("Empirical Coverage")
            ax.legend(frameon=True, fontsize=8)
            fig.tight_layout()
            save_figure(fig, output_dir, "v2_fig5_nominal_vs_empirical_coverage")

            if width_cols:
                fig, ax = plt.subplots(figsize=(7, 4.8))
                for estimator_key, grp in table_a.groupby("estimator_key", dropna=False):
                    ax.scatter(grp[width_cols[-1]], grp[cov_cols[-1]], s=30, label=estimator_key)
                ax.set_title("Width vs Coverage Frontier")
                ax.set_xlabel(width_cols[-1].replace("_", " ").title())
                ax.set_ylabel(cov_cols[-1].replace("_", " ").title())
                ax.legend(frameon=True, fontsize=8)
                fig.tight_layout()
                save_figure(fig, output_dir, "v2_fig6_width_vs_coverage_frontier")

        fig, ax = plt.subplots(figsize=(8, 4.8))
        estimators = sorted(point_df["estimator_key"].unique())
        box_data = [point_df.loc[point_df["estimator_key"] == est, "abs_error"].to_numpy() for est in estimators]
        ax.boxplot(box_data, tick_labels=estimators, showfliers=False)
        ax.set_title("Per-Estimator Absolute Error Distributions")
        ax.set_ylabel("Absolute Error")
        fig.tight_layout()
        save_figure(fig, output_dir, "v2_fig7_per_estimator_error_distributions")

    if not cross_ranking.empty:
        fig, ax = plt.subplots(figsize=(7, 4.8))
        ax.bar(cross_ranking["estimator_key"], cross_ranking["avg_rank_rmse"])
        ax.set_title("Cross-Estimator Ranking")
        ax.set_ylabel("Average RMSE Rank")
        fig.tight_layout()
        save_figure(fig, output_dir, "v2_fig8_cross_estimator_ranking")

    if not diagnostic_corr.empty:
        pivot = diagnostic_corr.pivot_table(index="estimator_key", columns="x_metric", values="correlation", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(8, 4.8))
        im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-1.0, vmax=1.0)
        ax.set_xticks(np.arange(len(pivot.columns)), labels=pivot.columns, rotation=30, ha="right")
        ax.set_yticks(np.arange(len(pivot.index)), labels=pivot.index)
        ax.set_title("Diagnostic Correlation Heatmap")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        save_figure(fig, output_dir, "v2_fig9_diagnostic_correlation_heatmap")

    if not table_b.empty:
        cols = [col for col in ["corr_ess_abs_error", "corr_ci_width_abs_error", "corr_ci_width_variance"] if col in table_b.columns]
        if cols:
            fig, ax = plt.subplots(figsize=(8, 4.8))
            x = np.arange(len(table_b))
            width = 0.25
            for idx, col in enumerate(cols):
                ax.bar(x + idx * width, table_b[col].fillna(0.0), width=width, label=col)
            ax.set_xticks(x + width, labels=table_b["estimator_key"], rotation=20, ha="right")
            ax.set_title("Diagnostic Quality Summary")
            ax.legend(frameon=True, fontsize=8)
            fig.tight_layout()
            save_figure(fig, output_dir, "v2_fig10_diagnostic_quality_bars")
