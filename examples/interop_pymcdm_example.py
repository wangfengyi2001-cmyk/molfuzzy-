"""
Interoperability example: use MolFuzzy's cognitive-map criterion weights
and aggregated decision matrix as input to a *different* MCDM library
(pymcdm's TOPSIS), and compare the resulting crisp ranking with
MolFuzzy's own MFS-ACO ranking on the same problem.

This demonstrates that MolFuzzy's intermediate artefacts (criterion
weights, aggregated decision matrix) are plain NumPy-compatible objects
that slot into other Python MCDM tooling, rather than being locked
inside a single monolithic pipeline.

Run with:  python examples/interop_pymcdm_example.py
Requires:  pip install pymcdm
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from molfuzzy import DecisionPipeline
from molfuzzy.data import aggregate_rectangular_mfv, linguistic_rect_matrix_to_mfv
from lean_energy_case_study import (
    ALTERNATIVES,
    CRITERIA,
    CRITERIA_MATRICES,
    DECISION_MATRICES,
)


def defuzzify_score(mfv) -> float:
    """Simple (m - n) score function mapped to [0, 1] for use as a crisp
    performance value in a conventional (non-fuzzy) MCDM method."""
    return (mfv.m - mfv.n + 1.0) / 2.0


def main() -> None:
    pipeline = DecisionPipeline(criterion_labels=CRITERIA, alternative_labels=ALTERNATIVES)
    result = pipeline.run(CRITERIA_MATRICES, DECISION_MATRICES)

    weights = result.criteria.weights
    print("MolFuzzy cognitive-map criterion weights:")
    for label, w in zip(CRITERIA, weights):
        print(f"  {label}: {w:.4f}")

    # Defuzzify MolFuzzy's aggregated decision matrix into a crisp
    # performance matrix for a conventional MCDM method.
    decision_mfv_matrices = [linguistic_rect_matrix_to_mfv(m) for m in DECISION_MATRICES]
    aggregated_decision = aggregate_rectangular_mfv(decision_mfv_matrices)
    crisp_matrix = np.array(
        [[defuzzify_score(cell) for cell in row] for row in aggregated_decision]
    )

    try:
        from pymcdm.methods import TOPSIS
    except ImportError:
        print("\n(pymcdm not installed -- skipping cross-library comparison; "
              "install with `pip install pymcdm` to run it)")
        return

    types = np.ones(len(CRITERIA))  # all four criteria are benefit criteria
    topsis = TOPSIS()
    scores = topsis(crisp_matrix, weights, types)
    order = np.argsort(-scores)

    print("\nTOPSIS (pymcdm) ranking using MolFuzzy-derived weights:")
    print("  " + " > ".join(ALTERNATIVES[i] for i in order))

    print("\nMolFuzzy MFS-ACO ranking (path-based, same problem):")
    print("  " + " > ".join(result.ranking.ranking_labels))


if __name__ == "__main__":
    main()
