from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ess_ope.data.dataset import EpisodeDataset
from ess_ope.estimators.dm_fqe import FQEResult, fitted_q_evaluation
from ess_ope.estimators.dr import doubly_robust_estimate
from ess_ope.estimators import compute_importance_weights
from ess_ope.policies.tabular import TabularPolicy


@dataclass
class MRDRResult:
    value: float
    fqe_result: FQEResult


def mrdr_linear_estimate(
    dataset: EpisodeDataset,
    target_policy: TabularPolicy,
    behavior_policy: TabularPolicy,
    feature_tensor: np.ndarray,
    num_states: int,
    num_actions: int,
    horizon: int,
    initial_state_dist: np.ndarray,
    gamma: float = 1.0,
    l2_reg: float = 1e-3,
    max_regression_weight: float = 1e3,
) -> MRDRResult:
    """MRDR-inspired weighted linear FQE + DR correction."""
    is_weights = compute_importance_weights(dataset, target_policy, behavior_policy)

    regression_weights = np.ones_like(is_weights.partial_weights)
    regression_weights[:, 1:] = is_weights.partial_weights[:, :-1]
    regression_weights = np.clip(regression_weights, 0.0, max_regression_weight)

    fqe_res = fitted_q_evaluation(
        dataset=dataset,
        target_policy=target_policy,
        num_states=num_states,
        num_actions=num_actions,
        horizon=horizon,
        initial_state_dist=initial_state_dist,
        model_type="linear",
        feature_tensor=feature_tensor,
        gamma=gamma,
        l2_reg=l2_reg,
        regression_weights=regression_weights,
    )

    value = doubly_robust_estimate(
        dataset=dataset,
        target_policy=target_policy,
        behavior_policy=behavior_policy,
        q_hat=fqe_res.q,
        v_hat=fqe_res.v,
        gamma=gamma,
    )

    return MRDRResult(value=value, fqe_result=fqe_res)
