from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm


@dataclass
class SummaryConfig:
    ci_level: float = 0.95
    bootstrap_samples: int = 400
    bootstrap_max_points: int = 5000
    bootstrap_batch_size: int = 32
    random_seed: int = 0
    show_progress: bool = False


@dataclass
class PaperClaimConfig:
    weak_corr_abs_max: float = 0.15
    same_ess_cv_max: float = 0.12
    same_ess_error_rel_min: float = 0.40
    same_ess_error_rel_strong: float = 0.75
    ess_change_ratio_min: float = 3.0
    ess_change_ratio_strong: float = 5.0
    stable_error_rel_max: float = 0.35
    stable_error_rel_strong: float = 0.25


ESTIMATOR_LABEL_TO_KEY = {
    "IS-PDIS": "is_pdis",
    "DM": "dm_tabular",
    "FQE": "fqe_linear",
    "MRDR": "mrdr",
}

ESTIMATOR_KEY_TO_LABEL = {v: k for k, v in ESTIMATOR_LABEL_TO_KEY.items()}


def _default_estimator_keys(df: pd.DataFrame) -> List[str]:
    keys = []
    for col in df.columns:
        if col.startswith("abs_error_"):
            key = col.replace("abs_error_", "")
            if f"estimate_{key}" in df.columns and f"error_{key}" in df.columns:
                keys.append(key)
    return sorted(keys)


def _mean_ci(values: np.ndarray, ci_level: float) -> tuple[float, float, float, float]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan

    mean = float(np.mean(v))
    std = float(np.std(v, ddof=1)) if n > 1 else 0.0
    se = std / np.sqrt(n) if n > 1 else 0.0
    z = float(stats.norm.ppf(0.5 + ci_level / 2.0))
    return mean, std, mean - z * se, mean + z * se


def _corr(x: np.ndarray, y: np.ndarray, method: str) -> float:
    if method == "pearson":
        if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
            return np.nan
        return float(np.corrcoef(x, y)[0, 1])
    if method == "spearman":
        corr = stats.spearmanr(x, y).correlation
        return float(corr)
    raise ValueError(f"Unsupported method: {method}")


def _bootstrap_corr_ci(
    x: np.ndarray,
    y: np.ndarray,
    method: str,
    n_boot: int,
    max_points: int,
    batch_size: int,
    ci_level: float,
    seed: int,
) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 4:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)
    if len(x) > max_points:
        idx = rng.choice(len(x), size=max_points, replace=False)
        x = x[idx]
        y = y[idx]

    point = _corr(x, y, method=method)
    if not np.isfinite(point):
        return np.nan, np.nan, np.nan

    # Use chunked vectorized bootstrap to reduce Python-loop overhead.
    # For Spearman bootstrap, compute Pearson on pre-ranked vectors for speed.
    if method == "pearson":
        x_work = x.astype(float, copy=False)
        y_work = y.astype(float, copy=False)
    elif method == "spearman":
        x_work = stats.rankdata(x).astype(float, copy=False)
        y_work = stats.rankdata(y).astype(float, copy=False)
    else:
        raise ValueError(f"Unsupported method: {method}")

    def _corr_rows(xb: np.ndarray, yb: np.ndarray) -> np.ndarray:
        x_mean = xb.mean(axis=1, keepdims=True)
        y_mean = yb.mean(axis=1, keepdims=True)
        xc = xb - x_mean
        yc = yb - y_mean
        cov = np.sum(xc * yc, axis=1)
        x_var = np.sum(xc * xc, axis=1)
        y_var = np.sum(yc * yc, axis=1)
        denom = np.sqrt(x_var * y_var)
        out = np.full(xb.shape[0], np.nan, dtype=float)
        mask = denom > 1e-20
        out[mask] = cov[mask] / denom[mask]
        return out

    boots: List[float] = []
    n = len(x_work)
    batch_size = max(1, int(batch_size))
    remaining = int(n_boot)
    while remaining > 0:
        b = min(batch_size, remaining)
        bidx = rng.integers(0, n, size=(b, n))
        vals = _corr_rows(x_work[bidx], y_work[bidx])
        vals = vals[np.isfinite(vals)]
        if vals.size:
            boots.extend(vals.tolist())
        remaining -= b

    if not boots:
        return point, np.nan, np.nan

    alpha = 1.0 - ci_level
    low = float(np.quantile(boots, alpha / 2.0))
    high = float(np.quantile(boots, 1.0 - alpha / 2.0))
    return point, low, high


def build_estimator_summary(
    df: pd.DataFrame,
    estimator_keys: Sequence[str] | None = None,
    config: SummaryConfig | None = None,
) -> pd.DataFrame:
    cfg = config or SummaryConfig()
    keys = list(estimator_keys) if estimator_keys is not None else _default_estimator_keys(df)

    rows = []
    key_iter = (
        tqdm(keys, desc="Estimator summary", leave=False)
        if cfg.show_progress
        else keys
    )
    for i, key in enumerate(key_iter):
        abs_err = df[f"abs_error_{key}"].to_numpy()
        sq_err = df[f"squared_error_{key}"].to_numpy()
        bias = df[f"error_{key}"].to_numpy()
        ess = df["ess_is_over_k"].to_numpy() if "ess_is_over_k" in df else (df["ess_is"] / df["K"].clip(lower=1)).to_numpy()

        mean_abs, std_abs, abs_low, abs_high = _mean_ci(abs_err, cfg.ci_level)
        mean_sq, _, sq_low, sq_high = _mean_ci(sq_err, cfg.ci_level)
        mean_bias, _, bias_low, bias_high = _mean_ci(bias, cfg.ci_level)

        pearson, pearson_low, pearson_high = _bootstrap_corr_ci(
            ess,
            abs_err,
            method="pearson",
            n_boot=cfg.bootstrap_samples,
            max_points=cfg.bootstrap_max_points,
            batch_size=cfg.bootstrap_batch_size,
            ci_level=cfg.ci_level,
            seed=cfg.random_seed + i * 17 + 1,
        )
        spearman, spearman_low, spearman_high = _bootstrap_corr_ci(
            ess,
            abs_err,
            method="spearman",
            n_boot=cfg.bootstrap_samples,
            max_points=cfg.bootstrap_max_points,
            batch_size=cfg.bootstrap_batch_size,
            ci_level=cfg.ci_level,
            seed=cfg.random_seed + i * 17 + 2,
        )

        rows.append(
            {
                "estimator": key,
                "n": int(len(abs_err)),
                "mean_abs_error": mean_abs,
                "std_abs_error": std_abs,
                "mean_abs_error_ci_low": abs_low,
                "mean_abs_error_ci_high": abs_high,
                "mean_squared_error": mean_sq,
                "mean_squared_error_ci_low": sq_low,
                "mean_squared_error_ci_high": sq_high,
                "mean_bias": mean_bias,
                "mean_bias_ci_low": bias_low,
                "mean_bias_ci_high": bias_high,
                "pearson_ess_abs_error": pearson,
                "pearson_ci_low": pearson_low,
                "pearson_ci_high": pearson_high,
                "spearman_ess_abs_error": spearman,
                "spearman_ci_low": spearman_low,
                "spearman_ci_high": spearman_high,
            }
        )

    return pd.DataFrame(rows)


def build_condition_summary(
    df: pd.DataFrame,
    estimator_keys: Sequence[str] | None = None,
    group_cols: Iterable[str] = ("alpha", "beta", "K"),
    ci_level: float = 0.95,
    show_progress: bool = False,
) -> pd.DataFrame:
    keys = list(estimator_keys) if estimator_keys is not None else _default_estimator_keys(df)
    gcols = list(group_cols)

    ess_norm = df["ess_is_over_k"] if "ess_is_over_k" in df else (df["ess_is"] / df["K"].clip(lower=1))
    base = df.assign(_ess_norm=ess_norm)
    ess_group = (
        base.groupby(gcols, as_index=False)
        .agg(
            mean_ess_is=("ess_is", "mean"),
            median_ess_is=("ess_is", "median"),
            mean_ess_norm=("_ess_norm", "mean"),
        )
    )

    z = float(stats.norm.ppf(0.5 + ci_level / 2.0))
    frames: List[pd.DataFrame] = []
    key_iter = (
        tqdm(keys, desc="Condition summary", leave=False)
        if show_progress
        else keys
    )
    for key in key_iter:
        err_col = f"abs_error_{key}"
        agg = (
            base.groupby(gcols)[err_col]
            .agg(n="size", mean_abs_error="mean", std_abs_error="std", median_abs_error="median")
            .reset_index()
        )
        agg["std_abs_error"] = agg["std_abs_error"].fillna(0.0)
        se = agg["std_abs_error"] / np.sqrt(agg["n"].clip(lower=1))
        agg["mean_abs_error_ci_low"] = agg["mean_abs_error"] - z * se
        agg["mean_abs_error_ci_high"] = agg["mean_abs_error"] + z * se
        agg["estimator"] = key
        agg = agg.merge(ess_group, on=gcols, how="left")
        frames.append(agg)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    sort_cols = [c for c in ["estimator", *gcols] if c in out.columns]
    return out.sort_values(sort_cols).reset_index(drop=True)


def _ci_confidence(ci_low: float, ci_high: float) -> str:
    if not np.isfinite(ci_low) or not np.isfinite(ci_high):
        return "low"
    width = ci_high - ci_low
    if width <= 0.15:
        return "high"
    if width <= 0.30:
        return "medium"
    return "low"


def _get_estimator_row(estimator_summary: pd.DataFrame, key: str) -> pd.Series | None:
    sub = estimator_summary[estimator_summary["estimator"] == key]
    if sub.empty:
        return None
    return sub.iloc[0]


def _get_benchmark_row(benchmark_report: pd.DataFrame, label: str) -> pd.Series | None:
    sub = benchmark_report[benchmark_report["estimator"] == label]
    if sub.empty:
        return None
    return sub.iloc[0]


def _ess_change_ratio(df: pd.DataFrame, beta_fixed: float) -> float:
    sub = df[np.isclose(df["beta"], beta_fixed)]
    if sub.empty:
        return np.nan
    by_alpha = sub.groupby("alpha")["ess_is"].median()
    if by_alpha.empty:
        return np.nan
    min_val = float(np.min(by_alpha.values))
    max_val = float(np.max(by_alpha.values))
    if min_val <= 0:
        return np.nan
    return max_val / min_val


def build_paper_claims_table(
    df: pd.DataFrame,
    estimator_summary: pd.DataFrame,
    benchmark_report: pd.DataFrame,
    config: PaperClaimConfig | None = None,
) -> pd.DataFrame:
    cfg = config or PaperClaimConfig()
    rows: List[dict] = []

    # Claim 1: ESS should be informative for IS.
    is_row = _get_estimator_row(estimator_summary, "is_pdis")
    if is_row is not None:
        spearman = float(is_row["spearman_ess_abs_error"])
        ci_low = float(is_row["spearman_ci_low"])
        ci_high = float(is_row["spearman_ci_high"])
        if np.isfinite(ci_high) and ci_high < -0.2:
            verdict = "supported"
        elif np.isfinite(ci_high) and ci_high < 0.0:
            verdict = "partially_supported"
        elif np.isfinite(ci_low) and np.isfinite(ci_high) and ci_low <= 0.0 <= ci_high:
            verdict = "inconclusive"
        else:
            verdict = "not_supported"

        rows.append(
            {
                "claim_id": "C1_IS_ESS_INFORMATIVE",
                "estimator": "IS-PDIS",
                "statement": "ESS should negatively correlate with IS error.",
                "metric": "spearman(ESS_norm, abs_error)",
                "value": spearman,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "threshold": "ci_high < 0 (strong: < -0.2)",
                "verdict": verdict,
                "confidence": _ci_confidence(ci_low, ci_high),
            }
        )

    # Claim 2: ESS weakly predicts non-IS errors (DM/FQE/MRDR).
    for key in ["dm_tabular", "fqe_linear", "mrdr"]:
        est = _get_estimator_row(estimator_summary, key)
        if est is None:
            continue
        label = ESTIMATOR_KEY_TO_LABEL.get(key, key)
        spearman = float(est["spearman_ess_abs_error"])
        ci_low = float(est["spearman_ci_low"])
        ci_high = float(est["spearman_ci_high"])
        abs_corr = abs(spearman)

        ci_abs_max = np.nanmax(np.abs([ci_low, ci_high]))
        contains_zero = np.isfinite(ci_low) and np.isfinite(ci_high) and ci_low <= 0.0 <= ci_high

        if abs_corr <= cfg.weak_corr_abs_max and np.isfinite(ci_abs_max) and ci_abs_max <= cfg.weak_corr_abs_max * 1.25:
            verdict = "supported"
        elif abs_corr <= cfg.weak_corr_abs_max * 1.5 and (contains_zero or ci_abs_max <= cfg.weak_corr_abs_max * 1.75):
            verdict = "partially_supported"
        elif not contains_zero and abs_corr > cfg.weak_corr_abs_max:
            verdict = "not_supported"
        else:
            verdict = "inconclusive"

        rows.append(
            {
                "claim_id": f"C2_{label}_WEAK_ESS_CORR",
                "estimator": label,
                "statement": "ESS should be weak/non-diagnostic for non-IS estimator error.",
                "metric": "spearman(ESS_norm, abs_error)",
                "value": spearman,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "threshold": (
                    f"|corr| <= {cfg.weak_corr_abs_max} (primary effect-size threshold); "
                    "CI near zero preferred"
                ),
                "verdict": verdict,
                "confidence": _ci_confidence(ci_low, ci_high),
            }
        )

    # Claim 3: Same ESS can still imply different errors (counterexample at fixed alpha).
    for label in ["DM", "FQE", "MRDR"]:
        bench = _get_benchmark_row(benchmark_report, label)
        key = ESTIMATOR_LABEL_TO_KEY.get(label)
        est = _get_estimator_row(estimator_summary, key) if key is not None else None
        if bench is None or est is None:
            continue

        err_range = float(bench["error_range_over_beta_at_alpha_fixed"])
        ess_cv = float(bench["ess_median_cv_over_beta_at_alpha_fixed"])
        mean_abs = max(float(est["mean_abs_error"]), 1e-12)
        err_rel = err_range / mean_abs

        if ess_cv <= cfg.same_ess_cv_max and err_rel >= cfg.same_ess_error_rel_strong:
            verdict = "supported"
        elif ess_cv <= cfg.same_ess_cv_max and err_rel >= cfg.same_ess_error_rel_min:
            verdict = "partially_supported"
        elif ess_cv > cfg.same_ess_cv_max * 1.5:
            verdict = "not_supported"
        else:
            verdict = "inconclusive"

        rows.append(
            {
                "claim_id": f"C3_{label}_SAME_ESS_DIFF_ERROR",
                "estimator": label,
                "statement": "At fixed alpha (similar ESS), error can differ strongly across beta.",
                "metric": "error_range_over_beta / mean_abs_error",
                "value": err_rel,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "threshold": (
                    f"ESS_CV <= {cfg.same_ess_cv_max} and rel_error_range >= {cfg.same_ess_error_rel_min} "
                    f"(strong >= {cfg.same_ess_error_rel_strong})"
                ),
                "aux_metric_ess_cv": ess_cv,
                "aux_metric_alpha_fixed": float(bench["alpha_fixed_used"]),
                "verdict": verdict,
                "confidence": "medium",
            }
        )

    # Claim 4: ESS can shift heavily while non-IS error stays relatively stable at beta_fixed.
    for label in ["DM", "FQE", "MRDR"]:
        bench = _get_benchmark_row(benchmark_report, label)
        key = ESTIMATOR_LABEL_TO_KEY.get(label)
        est = _get_estimator_row(estimator_summary, key) if key is not None else None
        if bench is None or est is None:
            continue

        beta_fixed = float(bench["beta_fixed_used"])
        ess_ratio = _ess_change_ratio(df, beta_fixed=beta_fixed)
        err_range_alpha = float(bench["error_range_over_alpha_at_beta_fixed"])
        mean_abs = max(float(est["mean_abs_error"]), 1e-12)
        err_rel_alpha = err_range_alpha / mean_abs

        if ess_ratio >= cfg.ess_change_ratio_strong and err_rel_alpha <= cfg.stable_error_rel_strong:
            verdict = "supported"
        elif ess_ratio >= cfg.ess_change_ratio_min and err_rel_alpha <= cfg.stable_error_rel_max:
            verdict = "partially_supported"
        elif ess_ratio >= cfg.ess_change_ratio_min and err_rel_alpha > cfg.stable_error_rel_max:
            verdict = "not_supported"
        elif np.isfinite(ess_ratio) and ess_ratio < cfg.ess_change_ratio_min:
            verdict = "not_supported"
        else:
            verdict = "inconclusive"

        rows.append(
            {
                "claim_id": f"C4_{label}_ESS_SHIFT_ERROR_STABLE",
                "estimator": label,
                "statement": "Across alpha at fixed beta, ESS may vary strongly while error remains comparatively stable.",
                "metric": "error_range_over_alpha / mean_abs_error",
                "value": err_rel_alpha,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "threshold": (
                    f"ESS_ratio >= {cfg.ess_change_ratio_min} and rel_error_range <= {cfg.stable_error_rel_max} "
                    f"(strong: ratio >= {cfg.ess_change_ratio_strong}, rel <= {cfg.stable_error_rel_strong})"
                ),
                "aux_metric_ess_ratio": ess_ratio,
                "aux_metric_beta_fixed": beta_fixed,
                "verdict": verdict,
                "confidence": "medium",
            }
        )

    if not rows:
        return pd.DataFrame()

    order = {"supported": 0, "partially_supported": 1, "inconclusive": 2, "not_supported": 3}
    out = pd.DataFrame(rows)
    out["verdict_rank"] = out["verdict"].map(order).fillna(99)
    return out.sort_values(["verdict_rank", "claim_id"]).drop(columns=["verdict_rank"]).reset_index(drop=True)


def build_paper_claim_summary(claims_df: pd.DataFrame) -> pd.DataFrame:
    if claims_df.empty:
        return pd.DataFrame(columns=["verdict", "count", "fraction"])

    counts = claims_df["verdict"].value_counts(dropna=False)
    total = int(len(claims_df))
    out = pd.DataFrame(
        {
            "verdict": counts.index.astype(str),
            "count": counts.values.astype(int),
            "fraction": counts.values / max(1, total),
        }
    )
    return out
