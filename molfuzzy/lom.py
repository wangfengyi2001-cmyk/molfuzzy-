"""
Stage 1 -- Linear Optimization Method (LOM) expert weighting.

Implements Eq. 5-8 of the reference methodology: experts are scored on
three reliability dimensions (internal consistency C, evaluation
stability V, Q-learning suitability Q), combined into a linear
reliability score R_e, normalized to a target vector R_bar_e, and then
fitted by a goal-programming linear model that minimizes total deviation
between the expert-weight vector and the reliability target, subject to
the weights summing to one.

The three reliability sub-dimensions are themselves expert-elicited /
domain quantities in the source methodology and are not fully
specified there; :func:`compute_reliability_dimensions` provides a
transparent, documented default (row-wise consistency, split-half
stability, distance-to-consensus suitability) that callers are free to
override by supplying their own ``C, V, Q`` arrays directly to
:func:`lom_expert_weights`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from scipy.optimize import linprog


@dataclass
class ReliabilityDimensions:
    """Per-expert reliability dimensions C_e, V_e, Q_e (Eq. 5)."""

    consistency: np.ndarray  # C_e
    stability: np.ndarray  # V_e
    suitability: np.ndarray  # Q_e


def compute_reliability_dimensions(
    number_matrices: Sequence[np.ndarray],
) -> ReliabilityDimensions:
    """Default, documented operationalisation of C_e, V_e, Q_e from each
    expert's raw assessment-number matrix.

    * Consistency (C_e): 1 - (mean coefficient of variation of each row),
      i.e. how internally coherent an expert's pairwise judgements are.
    * Stability (V_e): 1 - (relative split-half difference) between the
      row sums computed from odd/even columns, i.e. how much the
      expert's matrix "wobbles" depending on which half of the
      comparisons is used.
    * Suitability (Q_e): 1 - (normalized distance to the cross-expert
      consensus matrix), i.e. how representative the expert is of the
      panel and therefore how safe a Q-learning benchmark they would be.

    All three are returned in [0, 1] (higher is better).
    """
    experts = [np.asarray(mat, dtype=float) for mat in number_matrices]
    n_experts = len(experts)
    consensus = np.mean(np.stack(experts, axis=0), axis=0)

    consistency = np.zeros(n_experts)
    stability = np.zeros(n_experts)
    suitability = np.zeros(n_experts)

    for e, mat in enumerate(experts):
        n = mat.shape[0]
        row_cv = []
        for i in range(n):
            row = np.delete(mat[i], i)
            if row.mean() > 0:
                row_cv.append(row.std() / row.mean())
        consistency[e] = 1.0 - min(1.0, float(np.mean(row_cv)) if row_cv else 0.0)

        odd_cols = mat[:, 1::2].sum(axis=1)
        even_cols = mat[:, 0::2].sum(axis=1)
        denom = odd_cols + even_cols
        denom = np.where(denom == 0, 1.0, denom)
        split_half_gap = np.mean(np.abs(odd_cols - even_cols) / denom)
        stability[e] = 1.0 - min(1.0, float(split_half_gap))

        dist = np.linalg.norm(mat - consensus) / (np.linalg.norm(consensus) + 1e-9)
        suitability[e] = 1.0 - min(1.0, float(dist))

    return ReliabilityDimensions(consistency, stability, suitability)


def lom_expert_weights(
    dims: ReliabilityDimensions,
    beta: Tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
) -> dict:
    """Solve the LOM goal-programming model (Eq. 5-8).

    Parameters
    ----------
    dims : ReliabilityDimensions
        Per-expert C_e, V_e, Q_e (see :func:`compute_reliability_dimensions`).
    beta : (beta1, beta2, beta3)
        Importance coefficients for consistency, stability and
        suitability; must be non-negative and sum to 1 (Eq. 5).

    Returns
    -------
    dict with keys:
        R           raw linear reliability scores R_e (Eq. 5)
        R_bar       normalized reliability targets (Eq. 6)
        weights     optimized expert weights w_e^opt (Eq. 7-8)
        benchmark   index of the expert with the maximum optimized weight
        objective   optimal objective value Z* (should be ~0)
    """
    b1, b2, b3 = beta
    if not np.isclose(b1 + b2 + b3, 1.0):
        raise ValueError("beta coefficients must sum to 1")
    if min(beta) < 0:
        raise ValueError("beta coefficients must be non-negative")

    R = b1 * dims.consistency + b2 * dims.stability + b3 * dims.suitability
    R_bar = R / R.sum()

    n = len(R_bar)
    # Decision vector: [w_1..w_n, p_1..p_n, n_1..n_n]
    # minimize sum(p_e + n_e)
    c = np.concatenate([np.zeros(n), np.ones(n), np.ones(n)])

    # Equality constraints: w_e - p_e + n_e = R_bar_e  for each e
    # and sum(w_e) = 1
    A_eq = []
    b_eq = []
    for e in range(n):
        row = np.zeros(3 * n)
        row[e] = 1.0          # w_e
        row[n + e] = -1.0     # -p_e
        row[2 * n + e] = 1.0  # +n_e
        A_eq.append(row)
        b_eq.append(R_bar[e])
    sum_row = np.zeros(3 * n)
    sum_row[:n] = 1.0
    A_eq.append(sum_row)
    b_eq.append(1.0)

    bounds = [(0, None)] * (3 * n)

    result = linprog(
        c, A_eq=np.array(A_eq), b_eq=np.array(b_eq), bounds=bounds, method="highs"
    )
    if not result.success:
        raise RuntimeError(f"LOM linear program failed: {result.message}")

    weights = result.x[:n]
    benchmark = int(np.argmax(weights))

    return {
        "R": R,
        "R_bar": R_bar,
        "weights": weights,
        "benchmark": benchmark,
        "objective": result.fun,
    }
