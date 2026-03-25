from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd

from ess_ope.evaluation.summary import (
    ESTIMATOR_KEY_TO_LABEL,
    PaperClaimConfig,
    SummaryConfig,
    build_bias_variance_summary,
    build_chain_bandit_sensitivity_summary,
    build_ci_coverage_summary,
    build_ci_interval_summary,
    build_condition_summary,
    build_diagnostic_comparability_summary,
    build_estimator_summary,
    build_paper_claim_summary,
    build_paper_claims_table,
)
from ess_ope.plotting.benchmark_figures import generate_benchmark_figures
from ess_ope.utils.config import load_yaml
from ess_ope.utils.logging import sync_tracked_latest_snapshot


_RUN_DIR_RE = re.compile(r"^\d{8}_\d{6}_.+")
_VERDICT_ORDER = {
    "supported": 0,
    "partially_supported": 1,
    "inconclusive": 2,
    "not_supported": 3,
    "not_evaluated": 4,
}


def _infer_run_dir(output_dir: Path) -> Path:
    return output_dir.parent if output_dir.name == "figures" else output_dir


def _infer_results_root(path: Path) -> Path | None:
    for candidate in [path, *path.parents]:
        if candidate.name == "results":
            return candidate
    return None


def _safe_read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _aggregate_paper_claim_verdict(claims: pd.DataFrame) -> pd.DataFrame:
    if claims.empty:
        return pd.DataFrame(columns=["estimator", "paper_claim_verdict", "paper_claim_count"])

    rows = []
    for estimator, group in claims.groupby("estimator", dropna=False):
        verdicts = [str(value) for value in group["verdict"].dropna().tolist()]
        if verdicts:
            verdict = max(verdicts, key=lambda item: _VERDICT_ORDER.get(item, 99))
        else:
            verdict = "not_evaluated"
        rows.append(
            {
                "estimator": estimator,
                "paper_claim_verdict": verdict,
                "paper_claim_count": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def _strength_label(row: pd.Series) -> str:
    verdict = str(row.get("paper_claim_verdict", "not_evaluated"))
    coverage_gap = row.get("bootstrap_coverage_gap", np.nan)
    if verdict == "supported" and np.isfinite(coverage_gap) and abs(float(coverage_gap)) <= 0.05:
        return "strong"
    if verdict in {"supported", "partially_supported"}:
        return "mixed"
    if verdict == "inconclusive":
        return "watch"
    if verdict == "not_evaluated":
        return "unscored"
    return "weak"


def build_trial_scorecard(
    estimator_summary: pd.DataFrame,
    ci_coverage_summary: pd.DataFrame,
    paper_claims: pd.DataFrame,
) -> pd.DataFrame:
    if estimator_summary.empty:
        return pd.DataFrame()

    scorecard = estimator_summary.copy()
    scorecard["estimator_key"] = scorecard["estimator"].astype(str)
    scorecard["estimator"] = scorecard["estimator_key"].map(ESTIMATOR_KEY_TO_LABEL).fillna(scorecard["estimator_key"])

    coverage = ci_coverage_summary.copy()
    if not coverage.empty:
        coverage = coverage[coverage["ci_method"] == "bootstrap"].copy()
        coverage = coverage.rename(
            columns={
                "coverage_rate": "bootstrap_coverage_rate",
                "coverage_gap": "bootstrap_coverage_gap",
                "mean_width": "bootstrap_mean_width",
            }
        )
        keep = [
            "estimator",
            "bootstrap_coverage_rate",
            "bootstrap_coverage_gap",
            "bootstrap_mean_width",
        ]
        coverage = coverage[keep]

    claims = _aggregate_paper_claim_verdict(paper_claims)
    scorecard = scorecard.merge(coverage, on="estimator", how="left")
    scorecard = scorecard.merge(claims, on="estimator", how="left")
    scorecard["paper_claim_verdict"] = scorecard["paper_claim_verdict"].fillna("not_evaluated")
    scorecard["paper_claim_count"] = scorecard["paper_claim_count"].fillna(0).astype(int)
    scorecard["strength_label"] = scorecard.apply(_strength_label, axis=1)

    columns = [
        "estimator",
        "estimator_key",
        "mean_abs_error",
        "mean_bias",
        "spearman_ess_abs_error",
        "spearman_ci_low",
        "spearman_ci_high",
        "bootstrap_coverage_rate",
        "bootstrap_coverage_gap",
        "bootstrap_mean_width",
        "paper_claim_verdict",
        "paper_claim_count",
        "strength_label",
    ]
    out = scorecard[columns].copy()
    return out.sort_values("estimator").reset_index(drop=True)


def _row_count_for_run(run_dir: Path) -> float:
    csv_path = run_dir / "sweep_results.csv"
    parquet_path = run_dir / "sweep_results.parquet"
    if csv_path.exists():
        return float(len(pd.read_csv(csv_path)))
    if parquet_path.exists():
        return float(len(pd.read_parquet(parquet_path)))
    return np.nan


def _metric_from_scorecard(scorecard: pd.DataFrame, estimator: str, column: str) -> float:
    if scorecard.empty or column not in scorecard.columns:
        return np.nan
    sub = scorecard[scorecard["estimator"] == estimator]
    if sub.empty:
        return np.nan
    value = sub.iloc[0][column]
    return float(value) if pd.notna(value) else np.nan


def build_trial_index(results_root: str | Path) -> pd.DataFrame:
    root = Path(results_root)
    if not root.exists():
        return pd.DataFrame()

    rows = []
    for run_dir in sorted([path for path in root.iterdir() if path.is_dir() and _RUN_DIR_RE.match(path.name)]):
        figures_dir = run_dir / "figures"
        config = load_yaml(run_dir / "config.yaml") if (run_dir / "config.yaml").exists() else {}
        metadata = _safe_read_json(run_dir / "metadata.json")
        scorecard = _safe_read_csv(figures_dir / "trial_scorecard.csv")
        claim_summary = _safe_read_csv(figures_dir / "paper_claim_summary.csv")

        counts = {
            f"verdict_{verdict}_count": 0
            for verdict in ["supported", "partially_supported", "inconclusive", "not_supported"]
        }
        if not claim_summary.empty:
            for _, summary_row in claim_summary.iterrows():
                verdict = str(summary_row.get("verdict", ""))
                if verdict in counts:
                    counts[f"verdict_{verdict}_count"] = int(summary_row.get("count", 0))

        rows.append(
            {
                "run_id": run_dir.name,
                "env_name": config.get("env_name", "random_mdp"),
                "config_name": config.get("name", run_dir.name.split("_", 2)[-1]),
                "created_at_utc": metadata.get("created_at_utc", ""),
                "row_count": _row_count_for_run(run_dir),
                "has_chain_bandit_sensitivity": int((figures_dir / "chain_bandit_sensitivity_summary.csv").exists()),
                "is_spearman_ess_abs_error": _metric_from_scorecard(scorecard, "IS-PDIS", "spearman_ess_abs_error"),
                "dr_spearman_ess_abs_error": _metric_from_scorecard(scorecard, "DR", "spearman_ess_abs_error"),
                "fqe_spearman_ess_abs_error": _metric_from_scorecard(scorecard, "FQE", "spearman_ess_abs_error"),
                "is_bootstrap_coverage_rate": _metric_from_scorecard(scorecard, "IS-PDIS", "bootstrap_coverage_rate"),
                "dr_bootstrap_coverage_rate": _metric_from_scorecard(scorecard, "DR", "bootstrap_coverage_rate"),
                "fqe_bootstrap_coverage_rate": _metric_from_scorecard(scorecard, "FQE", "bootstrap_coverage_rate"),
                **counts,
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("run_id").reset_index(drop=True)


def _write_csvs(output_dir: Path, artifacts: Dict[str, pd.DataFrame], paper_mode: bool) -> None:
    for name, frame in artifacts.items():
        if name in {"paper_claims", "paper_claim_summary"} and not paper_mode:
            continue
        if name == "chain_bandit_sensitivity_summary" and frame.empty:
            continue
        frame.to_csv(output_dir / f"{name}.csv", index=False)


def _sync_latest_and_index(output_dir: Path, df: pd.DataFrame) -> None:
    run_dir = _infer_run_dir(output_dir)
    results_root = _infer_results_root(run_dir)
    if results_root is None or run_dir.parent != results_root:
        return

    env_values = [str(value) for value in sorted(df["env_name"].dropna().unique())] if "env_name" in df.columns else []
    if len(env_values) == 1:
        sync_tracked_latest_snapshot(results_root, run_dir, env_values[0])

    trial_index = build_trial_index(results_root)
    if not trial_index.empty:
        trial_index.to_csv(results_root / "trial_index.csv", index=False)


def generate_run_artifacts(
    df: pd.DataFrame,
    output_dir: str | Path,
    estimator_keys: Sequence[str] | None = None,
    fixed_alpha: float | None = None,
    fixed_beta: float = 0.0,
    bootstrap_samples: int = 16000,
    bootstrap_max_points: int = 200000,
    ci_level: float = 0.95,
    interval_methods: Sequence[str] | None = None,
    paper_mode: bool = True,
    progress: bool = False,
) -> Dict[str, pd.DataFrame]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary_cfg = SummaryConfig(
        ci_level=float(ci_level),
        bootstrap_samples=int(bootstrap_samples),
        bootstrap_max_points=int(bootstrap_max_points),
        show_progress=progress,
    )
    estimator_summary = build_estimator_summary(df, estimator_keys=estimator_keys, config=summary_cfg)
    condition_summary = build_condition_summary(
        df,
        estimator_keys=estimator_keys,
        ci_level=float(ci_level),
        show_progress=progress,
    )
    bias_variance_summary = build_bias_variance_summary(df, estimator_keys=estimator_keys)
    ci_interval_summary = build_ci_interval_summary(df, estimator_keys=estimator_keys, methods=interval_methods)
    ci_coverage_summary = build_ci_coverage_summary(ci_interval_summary, ci_level=float(ci_level))
    diagnostic_comparability = build_diagnostic_comparability_summary(
        df,
        ci_coverage_summary=ci_coverage_summary,
        estimator_keys=estimator_keys,
    )
    chain_bandit_sensitivity = build_chain_bandit_sensitivity_summary(df, estimator_keys=estimator_keys)
    benchmark_report = generate_benchmark_figures(
        df=df,
        output_dir=output_path,
        fixed_alpha=fixed_alpha,
        fixed_beta=fixed_beta,
        ci_level=float(ci_level),
        bias_variance_summary=bias_variance_summary,
        ci_interval_summary=ci_interval_summary,
        ci_coverage_summary=ci_coverage_summary,
        estimator_summary=estimator_summary,
        diagnostic_comparability=diagnostic_comparability,
    )
    paper_claims = build_paper_claims_table(
        df=df,
        estimator_summary=estimator_summary,
        benchmark_report=benchmark_report,
        config=PaperClaimConfig(),
    )
    paper_claim_summary = build_paper_claim_summary(paper_claims)
    trial_scorecard = build_trial_scorecard(estimator_summary, ci_coverage_summary, paper_claims)

    artifacts = {
        "benchmark_report": benchmark_report,
        "estimator_summary": estimator_summary,
        "condition_summary": condition_summary,
        "bias_variance_summary": bias_variance_summary,
        "ci_interval_summary": ci_interval_summary,
        "ci_coverage_summary": ci_coverage_summary,
        "diagnostic_comparability": diagnostic_comparability,
        "chain_bandit_sensitivity_summary": chain_bandit_sensitivity,
        "paper_claims": paper_claims,
        "paper_claim_summary": paper_claim_summary,
        "trial_scorecard": trial_scorecard,
    }
    _write_csvs(output_path, artifacts, paper_mode=paper_mode)
    _sync_latest_and_index(output_path, df)
    return artifacts
