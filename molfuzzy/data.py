"""Linguistic scale <-> assessment number <-> MFV conversions (Table 1)."""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from .mfv import MFV

# Linguistic term -> (assessment number, (m, n, h))
LINGUISTIC_SCALE: Dict[str, Dict[str, object]] = {
    "Negligible": {"number": 1, "mfv": (0.20, 0.70, 0.10)},
    "Low": {"number": 2, "mfv": (0.40, 0.50, 0.10)},
    "Moderate": {"number": 3, "mfv": (0.60, 0.30, 0.10)},
    "Significant": {"number": 4, "mfv": (0.80, 0.15, 0.05)},
    "High": {"number": 5, "mfv": (0.95, 0.05, 0.00)},
}

_ABBREV = {
    "N": "Negligible",
    "L": "Low",
    "M": "Moderate",
    "S": "Significant",
    "H": "High",
}


def term_to_mfv(term: str) -> MFV:
    """Look up a linguistic term (full name or single-letter code) and
    return its MFV triple, e.g. ``term_to_mfv("H")`` -> MFV(.95, .05, .00).
    """
    key = _ABBREV.get(term, term)
    if key not in LINGUISTIC_SCALE:
        raise KeyError(f"Unknown linguistic term '{term}'")
    return MFV(*LINGUISTIC_SCALE[key]["mfv"])


def term_to_number(term: str) -> int:
    key = _ABBREV.get(term, term)
    if key not in LINGUISTIC_SCALE:
        raise KeyError(f"Unknown linguistic term '{term}'")
    return int(LINGUISTIC_SCALE[key]["number"])


def assessment_to_mfv(number: int) -> MFV:
    """Reverse lookup: assessment number (1-5) -> MFV triple."""
    for entry in LINGUISTIC_SCALE.values():
        if entry["number"] == number:
            return MFV(*entry["mfv"])
    raise KeyError(f"No linguistic term with assessment number {number}")


def linguistic_matrix_to_numbers(
    labels: Sequence[str], matrix: Sequence[Sequence[str]]
) -> np.ndarray:
    """Convert a square matrix of linguistic codes (with an empty/None
    diagonal) into an assessment-number matrix. Diagonal entries are 0."""
    n = len(labels)
    out = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            cell = matrix[i][j]
            if cell in (None, "", "-"):
                continue
            out[i, j] = term_to_number(cell)
    return out


def normalized_relation_matrix(number_matrix: np.ndarray) -> np.ndarray:
    """Row-normalize an assessment-number matrix -- Eq. 4.

    r_ij = x_ij / sum_j x_ij  (row-wise), diagonal left at 0.
    """
    row_sums = number_matrix.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    normalized = number_matrix / row_sums
    np.fill_diagonal(normalized, 0.0)
    return normalized


def linguistic_rect_matrix_to_mfv(matrix: Sequence[Sequence[str]]) -> List[List[MFV]]:
    """Convert a rectangular (alternatives x criteria) linguistic matrix
    into an equally-shaped matrix of MFV triples. Public counterpart of
    the square-matrix converters above, useful when feeding MolFuzzy's
    decision matrices into other MCDM tooling."""
    return [[term_to_mfv(cell) for cell in row] for row in matrix]


def aggregate_rectangular_mfv(
    matrices: Sequence[List[List[MFV]]],
) -> List[List[MFV]]:
    """Elementwise mean (Eq. 13) of several rectangular MFV matrices, e.g.
    one per expert's decision matrix."""
    n_rows = len(matrices[0])
    n_cols = len(matrices[0][0])
    out: List[List[MFV]] = []
    for i in range(n_rows):
        row = []
        for j in range(n_cols):
            cell_values = [mat[i][j] for mat in matrices]
            row.append(MFV.average(cell_values))
        out.append(row)
    return out


def linguistic_matrix_to_mfv_rows(
    labels: Sequence[str], matrix: Sequence[Sequence[str]]
) -> List[List[MFV]]:
    """Convert a square linguistic matrix directly into rows of MFV
    (skipping the diagonal), i.e. the fuzzy vectors u_i of Eq. 14."""
    n = len(labels)
    rows: List[List[MFV]] = []
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                continue
            cell = matrix[i][j]
            row.append(term_to_mfv(cell) if cell not in (None, "", "-") else MFV(0, 0, 1))
        rows.append(row)
    return rows
