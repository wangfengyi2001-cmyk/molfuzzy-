"""
MolFuzzy: molecular-geometry-based fuzzy multi-criteria decision support.

MolFuzzy implements the pattern-based intelligent decision framework of
Molecular Fuzzy Values (MFV) combined with:

  * Linear Optimization Method (LOM) expert weighting
  * Q-learning based expert-evaluation balancing
  * Cognitive-map criteria weighting
  * Ant Colony Optimization (ACO) alternative ranking

as a reusable, domain-independent Python package. See ``molfuzzy.pipeline``
for the end-to-end orchestrator and ``examples/`` for a full worked case
study.

Public API
----------
MFV                     Molecular Fuzzy Value (membership, non-membership,
                         hesitancy) and molecular-geometry angle utilities.
GEOMETRIES               Names of the five supported molecular geometries.
lom_expert_weights        Linear-optimization expert weighting (Eq. 5-8).
balance_experts           Q-learning based evaluation-matrix balancing.
cognitive_map_weights     Cognitive-map criteria weighting.
aco_rank                  Ant-colony-optimization alternative ranking.
DecisionPipeline          High-level, 4-stage orchestrator (Fig. 1 of the
                         reference methodology): LOM expert weighting ->
                         Q-learning balancing -> cognitive-map criterion
                         weighting -> ACO alternative ranking.
"""

from .mfv import MFV, GEOMETRIES, angle_between, normalize_angle
from .lom import lom_expert_weights, ReliabilityDimensions
from .qlearning import balance_experts
from .cognitive_map import cognitive_map_weights
from .aco import aco_rank
from .pipeline import DecisionPipeline
from .data import (
    LINGUISTIC_SCALE,
    assessment_to_mfv,
    aggregate_rectangular_mfv,
    linguistic_rect_matrix_to_mfv,
)

__version__ = "1.0.0"

__all__ = [
    "MFV",
    "GEOMETRIES",
    "angle_between",
    "normalize_angle",
    "lom_expert_weights",
    "ReliabilityDimensions",
    "balance_experts",
    "cognitive_map_weights",
    "aco_rank",
    "DecisionPipeline",
    "LINGUISTIC_SCALE",
    "assessment_to_mfv",
    "aggregate_rectangular_mfv",
    "linguistic_rect_matrix_to_mfv",
    "__version__",
]
