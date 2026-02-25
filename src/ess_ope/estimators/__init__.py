"""Estimator namespace with keyword-safe re-exports."""

from importlib import import_module

_is_mod = import_module("ess_ope.estimators.is")

compute_importance_weights = _is_mod.compute_importance_weights
is_family_estimates = _is_mod.is_family_estimates
ISWeightResult = _is_mod.ISWeightResult

__all__ = [
    "compute_importance_weights",
    "is_family_estimates",
    "ISWeightResult",
]
