"""
DecisionPipeline: end-to-end orchestrator chaining all four stages.

Stage 1  Expert weighting            -- molfuzzy.lom
Stage 2  Evaluation balancing        -- molfuzzy.qlearning
Stage 3  Criterion weighting         -- molfuzzy.cognitive_map
Stage 4  Alternative ranking         -- molfuzzy.aco

This mirrors the four-column flowchart (Fig. 1) of the reference
methodology: qualified linguistic expert evaluations go in one end,
a ranked list of alternatives comes out the other, with every
intermediate artefact (expert weights, balanced matrices, criterion
weights, final decision/heuristic matrices, ant paths) exposed for
inspection or plotting.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

from .aco import ACOResult, aco_rank
from .cognitive_map import CognitiveMapResult, aggregate_matrices, cognitive_map_weights
from .data import (
    linguistic_matrix_to_mfv_rows,
    linguistic_matrix_to_numbers,
    normalized_relation_matrix,
)
from .lom import ReliabilityDimensions, compute_reliability_dimensions, lom_expert_weights
from .mfv import MFV
from .qlearning import QLearningResult, balance_experts


@dataclass
class PipelineResult:
    expert_weights: Dict
    balancing: QLearningResult
    criteria: CognitiveMapResult
    ranking: ACOResult
    criterion_labels: Sequence[str]
    alternative_labels: Sequence[str]

    def summary(self) -> str:
        lines = ["Criterion weights:"]
        for label, w in zip(self.criterion_labels, self.criteria.weights):
            lines.append(f"  {label}: {w:.4f}")
        lines.append("")
        lines.append(
            "Alternative ranking (best first): "
            + " > ".join(self.ranking.ranking_labels)
        )
        lines.append(f"Path fitness: {self.ranking.fitness:.4f}")
        return "\n".join(lines)


class DecisionPipeline:
    """Configure once, then call :meth:`run` with expert data.

    Example
    -------
    >>> pipeline = DecisionPipeline(
    ...     criterion_labels=["EE", "ADC", "EC", "PM"],
    ...     alternative_labels=["MNT", "JTM", "SMRT", "SCH", "AI"],
    ... )
    >>> result = pipeline.run(
    ...     criteria_matrices=[...],     # one linguistic matrix per expert
    ...     decision_matrices=[...],     # one linguistic matrix per expert
    ... )
    >>> print(result.summary())
    """

    def __init__(
        self,
        criterion_labels: Sequence[str],
        alternative_labels: Sequence[str],
        geometry: str = "linear",
        learning_rate: float = 0.1,
        beta: Sequence[float] = (1 / 3, 1 / 3, 1 / 3),
        aco_alpha: float = 1.0,
        aco_beta: float = 2.0,
        aco_rho: float = 0.5,
        aco_seed: int = 42,
    ):
        self.criterion_labels = list(criterion_labels)
        self.alternative_labels = list(alternative_labels)
        self.geometry = geometry
        self.learning_rate = learning_rate
        self.beta = beta
        self.aco_alpha = aco_alpha
        self.aco_beta = aco_beta
        self.aco_rho = aco_rho
        self.aco_seed = aco_seed

    def run(
        self,
        criteria_matrices: Sequence[Sequence[Sequence[str]]],
        decision_matrices: Sequence[Sequence[Sequence[str]]],
    ) -> PipelineResult:
        """Run all four stages.

        Parameters
        ----------
        criteria_matrices : one square linguistic matrix (criteria x
            criteria) per expert, used for LOM weighting, Q-learning
            balancing and cognitive-map criterion weighting.
        decision_matrices : one rectangular linguistic matrix
            (alternatives x criteria) per expert, used for the final ACO
            ranking once criterion weights are known.
        """
        # ---- Stage 1: LOM expert weighting (on the criteria matrices) ----
        number_matrices = [
            linguistic_matrix_to_numbers(self.criterion_labels, m)
            for m in criteria_matrices
        ]
        # Eq. 4 row-normalization; shares the single implementation in
        # ``molfuzzy.data`` so the two can never drift apart.
        norm_matrices = [normalized_relation_matrix(m) for m in number_matrices]
        dims = compute_reliability_dimensions(norm_matrices)
        lom = lom_expert_weights(dims, beta=self.beta)

        # ---- Stage 2: Q-learning evaluation balancing ----
        criteria_mfv_matrices = [
            _linguistic_square_to_mfv_matrix(self.criterion_labels, m)
            for m in criteria_matrices
        ]
        balancing = balance_experts(
            matrices=criteria_mfv_matrices,
            weights=lom["weights"],
            benchmark=lom["benchmark"],
            learning_rate=self.learning_rate,
        )

        # ---- Stage 3: Cognitive-map criterion weighting ----
        balanced_list = [balancing.balanced[i] for i in range(len(criteria_matrices))]
        aggregated_criteria = aggregate_matrices(balanced_list)
        cog = cognitive_map_weights(aggregated_criteria, geometry=self.geometry)

        # ---- Stage 4: ACO ranking (on the decision matrices) ----
        decision_mfv_matrices = [
            _linguistic_rect_to_mfv_matrix(m) for m in decision_matrices
        ]
        aggregated_decision = _aggregate_rectangular(decision_mfv_matrices)
        ranking = aco_rank(
            decision_matrix=aggregated_decision,
            weights=cog.weights,
            labels=self.alternative_labels,
            geometry=self.geometry,
            alpha=self.aco_alpha,
            beta=self.aco_beta,
            rho=self.aco_rho,
            seed=self.aco_seed,
        )

        return PipelineResult(
            expert_weights=lom,
            balancing=balancing,
            criteria=cog,
            ranking=ranking,
            criterion_labels=self.criterion_labels,
            alternative_labels=self.alternative_labels,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _linguistic_square_to_mfv_matrix(
    labels: Sequence[str], matrix: Sequence[Sequence[str]]
) -> List[List[MFV]]:
    from .data import term_to_mfv

    n = len(labels)
    out = []
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                row.append(MFV(0.0, 0.0, 1.0))
                continue
            row.append(term_to_mfv(matrix[i][j]))
        out.append(row)
    return out


def _linguistic_rect_to_mfv_matrix(matrix: Sequence[Sequence[str]]) -> List[List[MFV]]:
    from .data import linguistic_rect_matrix_to_mfv

    return linguistic_rect_matrix_to_mfv(matrix)


def _aggregate_rectangular(matrices: Sequence[List[List[MFV]]]) -> List[List[MFV]]:
    from .data import aggregate_rectangular_mfv

    return aggregate_rectangular_mfv(matrices)
