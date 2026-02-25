from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ess_ope.plotting.utils import save_figure


ESTIMATOR_SPECS: List[Tuple[str, str]] = [
    ("is_pdis", "IS-PDIS"),
    ("dm_tabular", "DM"),
    ("fqe_linear", "FQE"),
    ("mrdr", "MRDR"),
]


def _base_style() -> None:
    plt.style.use("seaborn-v0_8-white")
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 14,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "axes.linewidth": 1.7,
            "figure.facecolor": "white",
            "axes.facecolor": "#e8e8e8",
            "savefig.facecolor": "white",
        }
    )


def _style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("#e8e8e8")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.7)


def _x_ess_norm(df: pd.DataFrame) -> pd.Series:
    if "ess_is_over_k" in df:
        return df["ess_is_over_k"]
    return df["ess_is"] / df["K"].clip(lower=1)


def _available_estimators(df: pd.DataFrame, specs: Sequence[Tuple[str, str]]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for key, label in specs:
        if f"abs_error_{key}" in df and f"estimate_{key}" in df:
            out.append((key, label))
    return out


def _binned_median_line(x: np.ndarray, y: np.ndarray, bins: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    if len(x) < 10:
        return np.array([]), np.array([])

    x_min, x_max = float(np.min(x)), float(np.max(x))
    if np.isclose(x_min, x_max):
        return np.array([]), np.array([])

    edges = np.linspace(x_min, x_max, bins + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])
    med = np.full(bins, np.nan)

    for i in range(bins):
        mask = (x >= edges[i]) & (x < edges[i + 1] if i < bins - 1 else x <= edges[i + 1])
        if np.sum(mask) >= 5:
            med[i] = float(np.median(y[mask]))

    good = ~np.isnan(med)
    return mids[good], med[good]


def _corr_text(df: pd.DataFrame, x_col: str, y_col: str) -> str:
    pearson = float(df[x_col].corr(df[y_col], method="pearson"))
    spearman = float(df[x_col].corr(df[y_col], method="spearman"))
    return f"Pearson={pearson:.2f}\nSpearman={spearman:.2f}"


def _auto_select_fixed_alpha(df: pd.DataFrame, estimators: Sequence[Tuple[str, str]]) -> float:
    alpha_values = np.sort(df["alpha"].unique())
    if len(alpha_values) == 0:
        raise ValueError("No alpha values found in dataframe")

    candidates: List[Tuple[float, float, float]] = []

    for alpha in alpha_values:
        sub = df[np.isclose(df["alpha"], alpha)]
        if sub.empty:
            continue

        ess_by_beta = sub.groupby("beta")["ess_is"].median()
        if len(ess_by_beta) < 2 or float(ess_by_beta.mean()) <= 0:
            continue
        ess_cv = float(ess_by_beta.std() / ess_by_beta.mean())

        err_spreads: List[float] = []
        for key, _ in estimators:
            errs = sub.groupby("beta")[f"abs_error_{key}"].median()
            if len(errs) >= 2:
                err_spreads.append(float(errs.max() - errs.min()))

        if err_spreads:
            candidates.append((float(alpha), ess_cv, float(np.mean(err_spreads))))

    if not candidates:
        return float(alpha_values[len(alpha_values) // 2])

    min_cv = min(cv for _, cv, _ in candidates)
    cv_gate = max(0.10, min_cv + 0.03)
    preferred = [c for c in candidates if c[1] <= cv_gate]
    if not preferred:
        preferred = candidates

    preferred.sort(key=lambda x: (x[2], -x[1]), reverse=True)
    return preferred[0][0]


def _beta_colors(beta_values: np.ndarray) -> Dict[float, Tuple[float, float, float, float]]:
    vals = np.sort(np.unique(beta_values))
    cmap = plt.cm.viridis
    if len(vals) == 1:
        return {float(vals[0]): cmap(0.65)}
    return {float(v): cmap(i / (len(vals) - 1)) for i, v in enumerate(vals)}


def _fan_slice(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Select a slice where true value is near-constant for fan-style plot."""
    work = df.copy()
    max_k = int(work["K"].max())
    work = work[work["K"] == max_k]

    beta_values = np.sort(work["beta"].unique())
    beta_ref = float(beta_values[len(beta_values) // 2])
    work = work[np.isclose(work["beta"], beta_ref)]

    seed_counts = work.groupby("seed").size().sort_values(ascending=False)
    seed_ref = int(seed_counts.index[0])
    work = work[work["seed"] == seed_ref]

    if "env_repeat_id" in work.columns:
        env_counts = work.groupby("env_repeat_id").size().sort_values(ascending=False)
        env_ref = int(env_counts.index[0])
        work = work[work["env_repeat_id"] == env_ref]
    else:
        env_ref = 0

    if "policy_repeat_id" in work.columns:
        pol_counts = work.groupby("policy_repeat_id").size().sort_values(ascending=False)
        policy_ref = int(pol_counts.index[0])
        work = work[work["policy_repeat_id"] == policy_ref]
    else:
        policy_ref = 0

    truth_std = float(work["v_true"].std())
    info = {
        "beta_ref": beta_ref,
        "seed_ref": float(seed_ref),
        "env_repeat_ref": float(env_ref),
        "policy_repeat_ref": float(policy_ref),
        "K_ref": float(max_k),
        "truth_std": truth_std,
    }
    return work, info


def figure_1_ess_vs_error_by_estimator(df: pd.DataFrame, output_dir: str | Path) -> None:
    _base_style()
    estimators = _available_estimators(df, ESTIMATOR_SPECS)
    if not estimators:
        return

    plot_df = df.copy()
    plot_df["ess_norm"] = _x_ess_norm(plot_df)

    rows, cols = 2, 2
    fig, axes = plt.subplots(rows, cols, figsize=(13, 9), squeeze=False)
    colors = _beta_colors(plot_df["beta"].to_numpy())

    for i, (key, label) in enumerate(estimators):
        ax = axes[i // cols, i % cols]
        _style_axis(ax)
        y_col = f"abs_error_{key}"

        for beta in sorted(plot_df["beta"].unique()):
            sub = plot_df[np.isclose(plot_df["beta"], beta)]
            ax.scatter(
                sub["ess_norm"],
                sub[y_col],
                s=18,
                facecolors="none",
                edgecolors=colors[float(beta)],
                linewidths=1.0,
                alpha=0.6,
                label=f"beta={beta:g}",
            )

        bx, by = _binned_median_line(plot_df["ess_norm"].to_numpy(), plot_df[y_col].to_numpy())
        if len(bx) > 0:
            ax.plot(bx, by, color="black", linewidth=2.0)

        ax.text(
            0.98,
            0.97,
            _corr_text(plot_df, "ess_norm", y_col),
            ha="right",
            va="top",
            transform=ax.transAxes,
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85},
        )

        panel = chr(ord("A") + i)
        ax.set_title(f"{panel}. {label}")
        ax.set_xlabel("Normalized ESS (ESS / K)")
        ax.set_ylabel("Absolute Error")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles[: len(np.unique(df['beta']))], labels[: len(np.unique(df['beta']))], loc="upper center", ncol=4, frameon=True)

    fig.suptitle("ESS as reliability signal: IS vs DM/FQE/MRDR", y=0.99)
    fig.tight_layout(rect=[0.02, 0.03, 0.98, 0.95])
    save_figure(fig, output_dir, "benchmark_fig1_ess_vs_error_by_estimator")


def figure_2_same_ess_different_error(
    df: pd.DataFrame,
    output_dir: str | Path,
    fixed_alpha: Optional[float] = None,
) -> float:
    _base_style()
    estimators = _available_estimators(df, ESTIMATOR_SPECS)
    if not estimators:
        return float("nan")

    if fixed_alpha is None:
        fixed_alpha = _auto_select_fixed_alpha(df, estimators)

    sub = df[np.isclose(df["alpha"], fixed_alpha)].copy()
    if sub.empty:
        return float("nan")

    beta_values = np.sort(sub["beta"].unique())
    ess_meds = sub.groupby("beta")["ess_is"].median()
    ess_cv = float(ess_meds.std() / ess_meds.mean()) if float(ess_meds.mean()) > 0 else float("nan")

    rows, cols = 2, 2
    fig, axes = plt.subplots(rows, cols, figsize=(13, 9), squeeze=False)

    for i, (key, label) in enumerate(estimators):
        ax = axes[i // cols, i % cols]
        _style_axis(ax)
        y_col = f"abs_error_{key}"

        box_data = [sub[np.isclose(sub["beta"], beta)][y_col].to_numpy() for beta in beta_values]
        ax.boxplot(box_data, labels=[f"beta={b:g}" for b in beta_values], showfliers=False)

        y_top = ax.get_ylim()[1]
        for j, beta in enumerate(beta_values, start=1):
            ax.text(j, y_top * 0.97, f"ESS~{ess_meds.loc[beta]:.1f}", ha="center", va="top", fontsize=9)

        panel = chr(ord("A") + i)
        ax.set_title(f"{panel}. {label}")
        ax.set_ylabel("Absolute Error")

    fig.suptitle(
        f"Same-ESS counterexample (fixed alpha={fixed_alpha:g}, ESS median CV={ess_cv:.2f})",
        y=0.99,
    )
    fig.tight_layout(rect=[0.02, 0.03, 0.98, 0.95])
    save_figure(fig, output_dir, "benchmark_fig2_same_ess_different_error")
    return float(fixed_alpha)


def figure_3_ess_changes_error_stability(
    df: pd.DataFrame,
    output_dir: str | Path,
    fixed_beta: float = 0.0,
) -> None:
    _base_style()
    estimators = _available_estimators(df, ESTIMATOR_SPECS)
    if not estimators:
        return

    sub = df[np.isclose(df["beta"], fixed_beta)].copy()
    if sub.empty:
        return

    ess_by_alpha = sub.groupby("alpha", as_index=False).agg(ess_median=("ess_is", "median")).sort_values("alpha")

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(10, 8.5),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.35]},
    )
    _style_axis(ax_top)
    _style_axis(ax_bottom)

    ax_top.plot(ess_by_alpha["alpha"], ess_by_alpha["ess_median"], marker="o", color="#1f77b4", linewidth=2.0)
    ax_top.set_ylabel("Median IS-ESS")
    ax_top.set_title(f"A. ESS changes strongly with alpha (beta={fixed_beta:g})")

    colors = {
        "IS-PDIS": "#444444",
        "DM": "#1f77b4",
        "FQE": "#d62728",
        "MRDR": "#2ca02c",
    }
    for key, label in estimators:
        y_col = f"abs_error_{key}"
        g = sub.groupby("alpha", as_index=False).agg(err_median=(y_col, "median")).sort_values("alpha")
        ax_bottom.plot(
            g["alpha"],
            g["err_median"],
            marker="o",
            linewidth=2.0,
            label=label,
            color=colors.get(label, None),
        )

    ax_bottom.set_xlabel("alpha (policy divergence)")
    ax_bottom.set_ylabel("Median Absolute Error")
    ax_bottom.set_title("B. Error response by estimator")
    ax_bottom.legend(frameon=True, ncol=2)

    fig.tight_layout()
    save_figure(fig, output_dir, "benchmark_fig3_ess_changes_error_stability")


def figure_4_fan_estimate_vs_ess(df: pd.DataFrame, output_dir: str | Path) -> Dict[str, float]:
    _base_style()
    estimators = _available_estimators(df, ESTIMATOR_SPECS)
    if not estimators:
        return {}

    sub, slice_info = _fan_slice(df)
    if sub.empty:
        return {}

    rows, cols = 2, 2
    fig, axes = plt.subplots(rows, cols, figsize=(13, 9), squeeze=False)

    for i, (key, label) in enumerate(estimators):
        ax = axes[i // cols, i % cols]
        _style_axis(ax)

        x = sub["ess_is"].to_numpy()
        y = sub[f"estimate_{key}"].to_numpy()
        true_vals = sub["v_true"].to_numpy()

        ax.scatter(
            x,
            y,
            s=40,
            facecolors="none",
            edgecolors="#d62728",
            linewidths=1.4,
            alpha=0.35,
        )

        true_ref = float(np.mean(true_vals))
        ax.axhline(true_ref, linestyle="--", color="black", linewidth=2.8)

        panel = chr(ord("A") + i)
        ax.set_title(f"{panel}. {label}")
        ax.set_xlabel("IS-ESS")
        ax.set_ylabel("Policy Value Estimate")

    fig.suptitle(
        "Fan plot: estimate spread vs ESS (dashed = true value)",
        y=0.99,
    )
    fig.tight_layout(rect=[0.02, 0.03, 0.98, 0.95])
    save_figure(fig, output_dir, "benchmark_fig4_fan_estimate_vs_ess")

    return {
        "fan_beta": float(slice_info["beta_ref"]),
        "fan_seed": float(slice_info["seed_ref"]),
        "fan_K": float(slice_info["K_ref"]),
        "fan_true_std": float(slice_info["truth_std"]),
    }


def estimator_report(
    df: pd.DataFrame,
    fixed_alpha: Optional[float],
    fixed_beta: float,
) -> pd.DataFrame:
    estimators = _available_estimators(df, ESTIMATOR_SPECS)
    if not estimators:
        return pd.DataFrame()

    work = df.copy()
    work["ess_norm"] = _x_ess_norm(work)

    if fixed_alpha is None:
        fixed_alpha = _auto_select_fixed_alpha(work, estimators)

    rows: List[Dict[str, float | str]] = []
    for key, label in estimators:
        err_col = f"abs_error_{key}"
        row: Dict[str, float | str] = {
            "estimator": label,
            "pearson_all": float(work["ess_norm"].corr(work[err_col], method="pearson")),
            "spearman_all": float(work["ess_norm"].corr(work[err_col], method="spearman")),
            "alpha_fixed_used": float(fixed_alpha),
            "beta_fixed_used": float(fixed_beta),
        }

        alpha_sub = work[np.isclose(work["alpha"], fixed_alpha)]
        if not alpha_sub.empty:
            err_by_beta = alpha_sub.groupby("beta")[err_col].median()
            ess_by_beta = alpha_sub.groupby("beta")["ess_is"].median()
            if len(err_by_beta) >= 2:
                row["error_range_over_beta_at_alpha_fixed"] = float(err_by_beta.max() - err_by_beta.min())
            if len(ess_by_beta) >= 2 and float(ess_by_beta.mean()) > 0:
                row["ess_median_cv_over_beta_at_alpha_fixed"] = float(ess_by_beta.std() / ess_by_beta.mean())

        beta_sub = work[np.isclose(work["beta"], fixed_beta)]
        if not beta_sub.empty:
            err_by_alpha = beta_sub.groupby("alpha")[err_col].median()
            if len(err_by_alpha) >= 2:
                row["error_range_over_alpha_at_beta_fixed"] = float(err_by_alpha.max() - err_by_alpha.min())
            row["pearson_beta_fixed"] = float(beta_sub["ess_norm"].corr(beta_sub[err_col], method="pearson"))
            row["spearman_beta_fixed"] = float(beta_sub["ess_norm"].corr(beta_sub[err_col], method="spearman"))

        rows.append(row)

    return pd.DataFrame(rows)


def generate_benchmark_figures(
    df: pd.DataFrame,
    output_dir: str | Path,
    fixed_alpha: Optional[float] = None,
    fixed_beta: float = 0.0,
) -> pd.DataFrame:
    figure_1_ess_vs_error_by_estimator(df, output_dir)
    selected_alpha = figure_2_same_ess_different_error(df, output_dir, fixed_alpha=fixed_alpha)
    figure_3_ess_changes_error_stability(df, output_dir, fixed_beta=fixed_beta)
    fan_meta = figure_4_fan_estimate_vs_ess(df, output_dir)

    report = estimator_report(df, fixed_alpha=selected_alpha, fixed_beta=fixed_beta)
    if report.empty:
        return report

    for key, value in fan_meta.items():
        report[key] = value
    return report
