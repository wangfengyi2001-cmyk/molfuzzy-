"""
Stage 2 -- Q-learning based evaluation-matrix balancing.

Implements Eq. 9-12: the LOM-selected benchmark expert's MFV evaluation
matrix is treated as the reward reference Q_max, and every remaining
expert's matrix is pulled towards it through iterated reward/penalty
updates until the maximum elementwise change drops below 0.02.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .mfv import MFV

MatrixOfMFV = List[List[MFV]]


def _matrix_to_array(matrix: MatrixOfMFV) -> np.ndarray:
    """(n, n, 3) array view of a matrix of MFVs."""
    n = len(matrix)
    arr = np.zeros((n, n, 3))
    for i in range(n):
        for j in range(n):
            arr[i, j] = matrix[i][j].as_tuple()
    return arr


def _array_to_matrix(arr: np.ndarray) -> MatrixOfMFV:
    n = arr.shape[0]
    out: MatrixOfMFV = []
    for i in range(n):
        row = []
        for j in range(n):
            triple = arr[i, j]
            total = triple.sum()
            if total <= 0:
                row.append(MFV(0.0, 0.0, 1.0))
                continue
            row.append(MFV(*(triple / total)))
        out.append(row)
    return out


@dataclass
class QLearningResult:
    balanced: Dict[int, MatrixOfMFV]  # expert index -> balanced matrix
    benchmark: int
    iterations: int
    history: List[float] = field(default_factory=list)  # max|delta| per iteration


def balance_experts(
    matrices: Sequence[MatrixOfMFV],
    weights: Sequence[float],
    benchmark: int,
    learning_rate: float = 0.1,
    tolerance: float = 0.02,
    max_iterations: int = 100,
) -> QLearningResult:
    """Iteratively balance all non-benchmark expert matrices towards the
    LOM-selected benchmark expert (Eq. 9-12).

    Parameters
    ----------
    matrices : sequence of (n x n) MFV matrices, one per expert.
    weights : optimized LOM weights w_e^opt, same order as ``matrices``.
    benchmark : index of the benchmark expert (max-weight expert).
    learning_rate : lambda in Eq. 11 (paper default 0.1; also exercised
        at 0.5 and 1.0 in the sensitivity analysis).
    tolerance : convergence threshold on Eq. 12 (paper default 0.02).
    max_iterations : safety cap on the iteration count.
    """
    arrays = [_matrix_to_array(m) for m in matrices]
    q_max = arrays[benchmark].copy()

    balanced: Dict[int, np.ndarray] = {}
    history: List[float] = []
    iterations_used = 0

    for k, q_k in enumerate(arrays):
        if k == benchmark:
            balanced[k] = q_k.copy()
            continue

        w_k = weights[k]
        current = q_k.copy()
        for it in range(1, max_iterations + 1):
            reward = w_k * (q_max - current)          # Eq. 9
            penalty = np.maximum(w_k * (current - q_max), 0.0)  # Eq. 10
            updated = current + learning_rate * (reward - penalty)  # Eq. 11
            updated = np.clip(updated, 0.0, 1.0)
            # renormalize each (m, n, h) triple to sum to 1
            sums = updated.sum(axis=2, keepdims=True)
            sums = np.where(sums == 0, 1.0, sums)
            updated = updated / sums

            max_delta = float(np.max(np.abs(updated - current)))
            history.append(max_delta)
            current = updated
            iterations_used = max(iterations_used, it)
            if max_delta < tolerance:  # Eq. 12
                break

        balanced[k] = current

    balanced_matrices = {k: _array_to_matrix(v) for k, v in balanced.items()}
    return QLearningResult(
        balanced=balanced_matrices,
        benchmark=benchmark,
        iterations=iterations_used,
        history=history,
    )
