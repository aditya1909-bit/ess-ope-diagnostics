"""Paper-specific study pipeline for ESS vs uncertainty experiments."""

from ess_ope.study.config import ExperimentConfig, StudyConfig

__all__ = [
    "ExperimentConfig",
    "StudyConfig",
    "analyze_saved_results",
    "run_experiment",
    "run_study",
]


def run_experiment(*args, **kwargs):
    from ess_ope.study.runner import run_experiment as _run_experiment

    return _run_experiment(*args, **kwargs)


def run_study(*args, **kwargs):
    from ess_ope.study.runner import run_study as _run_study

    return _run_study(*args, **kwargs)


def analyze_saved_results(*args, **kwargs):
    from ess_ope.study.runner import analyze_saved_results as _analyze_saved_results

    return _analyze_saved_results(*args, **kwargs)
