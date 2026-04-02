from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from ess_ope.utils.config import load_yaml


def _default_phase_name(env_family: str) -> str:
    return f"{env_family}_v2"


@dataclass
class PhaseConfig:
    name: str = "ope_v2_phase"
    env_family: str = "contextual_bandit"
    results_root: str = "results"
    gamma: float = 1.0
    seeds: List[int] = field(default_factory=lambda: [0])
    replicates: int = 10
    sample_sizes: List[int] = field(default_factory=lambda: [100, 500, 2000])
    mismatch_levels: List[str] = field(default_factory=lambda: ["low", "medium", "high"])
    reward_noise_levels: List[str] = field(default_factory=lambda: ["low", "medium", "high"])
    support_regimes: List[str] = field(default_factory=lambda: ["full", "weak"])
    horizons: List[int] = field(default_factory=lambda: [1])
    reward_regimes: List[str] = field(default_factory=lambda: ["normal"])
    rarity_levels: List[str] = field(default_factory=lambda: ["none"])
    estimators: List[str] = field(default_factory=lambda: ["is", "wis", "dr"])
    ci_methods: List[str] = field(default_factory=lambda: ["analytic", "bootstrap_percentile", "bootstrap_basic"])
    ci_levels: List[float] = field(default_factory=lambda: [0.9, 0.95])
    bootstrap_samples: int = 200
    bootstrap_subsample_ratio: float = 1.0
    include_reference_estimators: bool = False
    num_workers: int = 1
    mp_chunksize: int = 1
    output_subdir: str = "artifacts"
    env: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PhaseConfig":
        base = cls()
        for key, value in payload.items():
            if hasattr(base, key):
                setattr(base, key, value)
        if not base.name:
            base.name = _default_phase_name(base.env_family)
        return base

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PhaseConfig":
        return cls.from_dict(load_yaml(path))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SuiteConfig:
    name: str = "ope_v2_suite"
    results_root: str = "results"
    phase_configs: List[str] = field(default_factory=list)
    output_subdir: str = "paper_artifacts"

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SuiteConfig":
        base = cls()
        for key, value in payload.items():
            if hasattr(base, key):
                setattr(base, key, value)
        return base

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SuiteConfig":
        return cls.from_dict(load_yaml(path))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
