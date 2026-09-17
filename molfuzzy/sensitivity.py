"""
Sensitivity-analysis helper: sweep the Q-learning rate and the molecular
geometry to check the stability of criterion weights and rankings (the
same experiment design as Section 4.4 / Table 25 of the reference
methodology).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from .mfv import GEOMETRIES
from .pipeline import DecisionPipeline, PipelineResult


@dataclass
class SensitivityRow:
    learning_rate: float
    geometry: str
    weights: np.ndarray
    ranking_order: List[str]


def run_sensitivity_analysis(
    criterion_labels: Sequence[str],
    alternative_labels: Sequence[str],
    criteria_matrices,
    decision_matrices,
    learning_rates: Sequence[float] = (0.1, 0.5, 1.0),
    geometries: Sequence[str] = GEOMETRIES,
    aco_seed: int = 42,
) -> List[SensitivityRow]:
    """Re-run the full pipeline for every (learning_rate, geometry) pair
    and collect the resulting criterion weights and their rank order.

    Returns one :class:`SensitivityRow` per scenario.

    Note that ``ranking_order`` is the rank order of the *criterion
    weights* (Stage 3), not of the alternatives; whether it is stable is
    an empirical outcome to be checked with :func:`rankings_are_stable`,
    not a guarantee. The ``weights`` arrays can also be used to plot a
    sensitivity chart such as the reference Fig. 4.
    """
    rows: List[SensitivityRow] = []
    for lr in learning_rates:
        for geometry in geometries:
            pipeline = DecisionPipeline(
                criterion_labels=criterion_labels,
                alternative_labels=alternative_labels,
                geometry=geometry,
                learning_rate=lr,
                aco_seed=aco_seed,
            )
            result: PipelineResult = pipeline.run(criteria_matrices, decision_matrices)
            order = [
                label
                for label, _ in sorted(
                    zip(criterion_labels, result.criteria.weights), key=lambda x: -x[1]
                )
            ]
            rows.append(
                SensitivityRow(
                    learning_rate=lr,
                    geometry=geometry,
                    weights=result.criteria.weights.copy(),
                    ranking_order=order,
                )
            )
    return rows


def rankings_are_stable(rows: Sequence[SensitivityRow]) -> bool:
    """True if every scenario produced the same criterion rank order."""
    orders = {tuple(r.ranking_order) for r in rows}
    return len(orders) == 1
