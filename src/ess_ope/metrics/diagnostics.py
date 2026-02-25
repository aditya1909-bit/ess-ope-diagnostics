from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd
from scipy import stats


def correlation_stats(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return {"pearson_r": np.nan, "pearson_p": np.nan, "spearman_r": np.nan, "spearman_p": np.nan}

    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_r, spearman_p = stats.spearmanr(x, y)
    return {
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
    }


def ess_error_correlations(
    df: pd.DataFrame,
    ess_column: str,
    error_columns: Iterable[str],
) -> pd.DataFrame:
    rows = []
    for col in error_columns:
        if col not in df:
            continue
        stats_row = correlation_stats(df[ess_column].to_numpy(), df[col].to_numpy())
        stats_row["error_column"] = col
        rows.append(stats_row)
    return pd.DataFrame(rows)
