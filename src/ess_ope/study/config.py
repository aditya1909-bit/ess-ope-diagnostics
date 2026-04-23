from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from ess_ope.utils.config import load_yaml


DEFAULT_ESTIMATORS = ["is", "snis", "pdis", "dm", "dr", "mrdr", "fqe"]


@dataclass
class ExperimentConfig:
    experiment_id: str = "experiment_1"
    title: str = ""
    results_root: str = "results"
    output_subdir: str = "artifacts"
    environment: Dict[str, Any] = field(default_factory=dict)
    grid: Dict[str, List[Any]] = field(default_factory=dict)
    estimators: List[str] = field(default_factory=lambda: list(DEFAULT_ESTIMATORS))
    intervals: Dict[str, Any] = field(
        default_factory=lambda: {
            "methods": ["bootstrap_percentile", "bootstrap_normal"],
            "levels": [0.9],
            "primary_level": 0.9,
            "bootstrap_samples": 500,
            "subsample_ratio": 1.0,
            "enable_concentration": False,
        }
    )
    replicates: int = 500
    seeds: List[int] = field(default_factory=lambda: [0])
    truth_method: str = "dynamic_programming"
    num_workers: int = -1
    mp_chunksize: int = 1

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ExperimentConfig":
        base = cls()
        for key, value in payload.items():
            if hasattr(base, key):
                setattr(base, key, value)
        return base

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        return cls.from_dict(load_yaml(path))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StudyConfig:
    name: str = "paper_study"
    results_root: str = "results"
    output_subdir: str = "artifacts"
    experiment_configs: List[str] = field(default_factory=list)
    overrides: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "StudyConfig":
        base = cls()
        for key, value in payload.items():
            if hasattr(base, key):
                setattr(base, key, value)
        return base

    @classmethod
    def from_yaml(cls, path: str | Path) -> "StudyConfig":
        return cls.from_dict(load_yaml(path))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def merge_experiment_overrides(config: ExperimentConfig, overrides: Dict[str, Any]) -> ExperimentConfig:
    payload = config.to_dict()
    for key, value in overrides.items():
        if key in {"environment", "grid", "intervals"} and isinstance(value, dict):
            merged = dict(payload.get(key, {}))
            merged.update(value)
            payload[key] = merged
        else:
            payload[key] = value
    return ExperimentConfig.from_dict(payload)
