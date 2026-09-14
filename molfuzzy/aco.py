"""
Stage 4 -- Ant Colony Optimization (ACO) alternative ranking under MFS.

Implements Eq. 22-28: criterion weights are folded into the aggregated
decision matrix to obtain weighted fuzzy vectors per alternative
(Eq. 22-23); pairwise angles between alternatives are turned into a
column-normalized final decision matrix and a heuristic-desirability
matrix (Eq. 24-25); a colony of ants (one per alternative starting
point) then builds full visiting sequences using the standard
pheromone/heuristic transition rule (Eq. 26), scores each sequence by
its path fitness (Eq. 27) and updates pheromone trails (Eq. 28) until
the best sequence stops changing between iterations. The best-fitness
sequence in the final iteration is returned as the alternative ranking.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .mfv import (
    MFV,
    WeightedTriple,
    build_angle_matrix,
    column_normalize,
    reciprocal_matrix,
)


@dataclass
class AntIterationLog:
    paths: List[List[int]]
    fitness: List[float]
    pheromone: np.ndarray


@dataclass
class ACOResult:
    ranking: List[int]              # alternative indices, best-first
    ranking_labels: List[str]
    fitness: float
    final_decision_matrix: np.ndarray
    heuristic_matrix: np.ndarray
    iterations: int
    log: List[AntIterationLog] = field(default_factory=list)


def _weighted_vectors(
    decision_matrix: List[List[MFV]], weights: Sequence[float]
) -> List[List[WeightedTriple]]:
    """Eq. 22-23: weight each MFV by its criterion weight and assemble the
    per-alternative fuzzy vector."""
    vectors = []
    for row in decision_matrix:
        vectors.append([mfv.scale(w) for mfv, w in zip(row, weights)])
    return vectors


def _final_decision_and_heuristic(
    vectors: List[List[WeightedTriple]], geometry: str
) -> Tuple[np.ndarray, np.ndarray]:
    angles = build_angle_matrix(vectors, geometry)      # Eq. 24
    recip = reciprocal_matrix(angles)                   # Eq. 17
    final_decision = column_normalize(recip)             # "final decision matrix" f
    with np.errstate(divide="ignore"):
        heuristic = np.where(final_decision > 0, 1.0 / final_decision, 0.0)  # Eq. 25
    np.fill_diagonal(heuristic, 0.0)
    return final_decision, heuristic


def _path_fitness(path: Sequence[int], final_decision: np.ndarray) -> float:
    total = sum(
        final_decision[path[i], path[i + 1]] for i in range(len(path) - 1)
    )
    return 1.0 / total if total > 0 else 0.0  # Eq. 27


def _build_ant_path(
    start: int,
    n: int,
    pheromone: np.ndarray,
    heuristic: np.ndarray,
    alpha: float,
    beta: float,
    rng: np.random.Generator,
) -> List[int]:
    visited = [start]
    unvisited = set(range(n)) - {start}
    current = start
    while unvisited:
        candidates = list(unvisited)
        scores = np.array(
            [
                (pheromone[current, j] ** alpha) * (heuristic[current, j] ** beta)
                for j in candidates
            ]
        )
        total = scores.sum()
        if total <= 0 or not np.isfinite(total):
            probs = np.ones(len(candidates)) / len(candidates)
        else:
            probs = scores / total  # Eq. 26
        next_city = rng.choice(candidates, p=probs)
        visited.append(int(next_city))
        unvisited.remove(next_city)
        current = next_city
    return visited


def aco_rank(
    decision_matrix: List[List[MFV]],
    weights: Sequence[float],
    labels: Sequence[str],
    geometry: str = "linear",
    alpha: float = 1.0,
    beta: float = 2.0,
    rho: float = 0.5,
    initial_pheromone: float = 0.1,
    max_iterations: int = 50,
    seed: int = 42,
) -> ACOResult:
    """Rank alternatives with the MFS-ACO stage (Eq. 22-28).

    Parameters
    ----------
    decision_matrix : (n_alternatives x n_criteria) aggregated MFV matrix.
    weights : criterion weights (from :func:`molfuzzy.cognitive_map.cognitive_map_weights`).
    labels : alternative names/labels, same order as ``decision_matrix`` rows.
    alpha, beta : pheromone / heuristic exponents in the transition rule
        (Eq. 26); not numerically fixed in the source methodology, default
        to the common ACO choice alpha=1, beta=2.
    rho : pheromone evaporation rate (paper default 0.5).
    initial_pheromone : tau(0) (paper default 0.1).
    seed : RNG seed; ants sample the transition rule stochastically, so a
        fixed seed makes a run reproducible.
    """
    n_criteria = len(weights)
    if any(len(row) != n_criteria for row in decision_matrix):
        raise ValueError("every decision_matrix row must have one MFV per criterion")
    if len(decision_matrix) != len(labels):
        raise ValueError("decision_matrix must have exactly one row per label")

    vectors = _weighted_vectors(decision_matrix, weights)
    final_decision, heuristic = _final_decision_and_heuristic(vectors, geometry)

    n_alt = len(labels)
    pheromone = np.full((n_alt, n_alt), initial_pheromone)
    rng = np.random.default_rng(seed)

    log: List[AntIterationLog] = []
    previous_best_path: List[int] | None = None
    best_path: List[int] = list(range(n_alt))
    best_fitness = -np.inf

    for iteration in range(1, max_iterations + 1):
        paths = [
            _build_ant_path(start, n_alt, pheromone, heuristic, alpha, beta, rng)
            for start in range(n_alt)
        ]
        fitness = [_path_fitness(p, final_decision) for p in paths]

        # Eq. 28: evaporate, then deposit along each ant's own path an
        # amount proportional to that ant's path fitness F(R_k) -- since
        # F(R_k) is already defined (Eq. 27) as the reciprocal of the
        # path's summed decision-matrix cost, higher fitness (shorter,
        # more attractive path) reinforces its edges more strongly, the
        # standard Ant-System convention.
        deposit = np.zeros((n_alt, n_alt))
        for path, fit in zip(paths, fitness):
            if fit <= 0:
                continue
            for i in range(len(path) - 1):
                deposit[path[i], path[i + 1]] += fit
        pheromone = (1 - rho) * pheromone + deposit

        log.append(AntIterationLog(paths=paths, fitness=fitness, pheromone=pheromone.copy()))

        iter_best_idx = int(np.argmax(fitness))
        iter_best_path = paths[iter_best_idx]
        iter_best_fitness = fitness[iter_best_idx]
        if iter_best_fitness > best_fitness:
            best_fitness = iter_best_fitness
            best_path = iter_best_path

        if previous_best_path is not None and iter_best_path == previous_best_path:
            break
        previous_best_path = iter_best_path

    ranking_labels = [labels[i] for i in best_path]
    return ACOResult(
        ranking=best_path,
        ranking_labels=ranking_labels,
        fitness=best_fitness,
        final_decision_matrix=final_decision,
        heuristic_matrix=heuristic,
        iterations=len(log),
        log=log,
    )
