"""
Stage 3 -- Cognitive-map criteria weighting under MFS.

Implements Eq. 13-21: the balanced expert matrices are aggregated into a
single fuzzy relation matrix, converted into an NR-matrix through the
molecular-geometry angle pipeline (see ``molfuzzy.mfv``), and used to
iterate a sigmoid state-vector update (a Kosko-style fuzzy-cognitive-map
activation rule) until convergence, at which point the stabilized state
vector is normalized into criterion weights.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

from .mfv import MFV, nr_matrix


@dataclass
class CognitiveMapResult:
    aggregated: List[List[MFV]]
    nr_matrix: np.ndarray
    state_vectors: List[np.ndarray]
    weights: np.ndarray
    iterations: int


def aggregate_matrices(matrices: Sequence[List[List[MFV]]]) -> List[List[MFV]]:
    """Eq. 13: elementwise mean of several balanced (n x n) MFV matrices."""
    n = len(matrices[0])
    aggregated: List[List[MFV]] = []
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                row.append(MFV(0.0, 0.0, 1.0))
                continue
            cell_values = [mat[i][j] for mat in matrices]
            row.append(MFV.average(cell_values))
        aggregated.append(row)
    return aggregated


def _rows_excluding_diagonal(matrix: List[List[MFV]]) -> List[List[MFV]]:
    """Eq. 14: build the (n-1)-dimensional fuzzy vector for each row by
    dropping the diagonal element."""
    n = len(matrix)
    rows = []
    for i in range(n):
        rows.append([matrix[i][j] for j in range(n) if j != i])
    return rows


def cognitive_map_weights(
    aggregated: List[List[MFV]],
    geometry: str = "linear",
    tolerance: float = 1e-4,
    max_iterations: int = 200,
) -> CognitiveMapResult:
    """Run the full cognitive-map pipeline (Eq. 14-21) on an already
    aggregated (Eq. 13) fuzzy relation matrix and return criterion
    weights.

    Parameters
    ----------
    aggregated : n x n MFV relation matrix (criteria x criteria).
    geometry : one of ``molfuzzy.mfv.GEOMETRIES``.
    tolerance : convergence tolerance for Eq. 20 (paper uses exact
        iteration equality; a small numeric tolerance is used here for
        floating-point robustness).
    """
    n = len(aggregated)
    rows = _rows_excluding_diagonal(aggregated)
    nr = nr_matrix(rows, geometry)  # Eq. 15-17

    state = np.ones(n)
    history = [state.copy()]
    iterations = 0
    for it in range(1, max_iterations + 1):
        raw = nr @ state
        updated = 1.0 / (1.0 + np.exp(-raw))  # Eq. 19 (sigmoid activation)
        history.append(updated.copy())
        iterations = it
        if np.max(np.abs(updated - state)) < tolerance:  # Eq. 20
            state = updated
            break
        state = updated

    weights = state / state.sum()  # Eq. 21
    return CognitiveMapResult(
        aggregated=aggregated,
        nr_matrix=nr,
        state_vectors=history,
        weights=weights,
        iterations=iterations,
    )
