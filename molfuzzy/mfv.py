"""
Molecular Fuzzy Values (MFV) and molecular-geometry angle utilities.

A Molecular Fuzzy Value is a triple (m, n, h) of membership,
non-membership and hesitancy degrees with m + n + h = 1, whose angular
relationships are interpreted with respect to a molecular geometry
(linear, trigonal planar, tetrahedral, trigonal bipyramidal, octahedral).
This module implements:

  * the MFV container with validation and elementary operations,
  * the membership / non-membership / hesitancy degree equations
    (Eq. 1-3 of the reference methodology),
  * cosine-similarity angles between fuzzy vectors built from MFV rows
    (Eq. 15 / Eq. 24), and
  * the geometry-dependent piecewise angle normalization and reciprocal
    transform used by both the cognitive-map and ACO stages
    (Eq. 16-17).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import acos, cos, pi, sqrt
from typing import Iterable, List, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Molecular geometries and their characteristic (maximum) bond angle, used as
# the normalizing denominator in Eq. 16. The mapping is a configurable model
# parameter: callers may register additional geometries via
# ``register_geometry`` without modifying this module.
# ---------------------------------------------------------------------------
GEOMETRIES = (
    "linear",
    "trigonal_planar",
    "tetrahedral",
    "trigonal_bipyramidal",
    "octahedral",
)

_GEOMETRY_DENOMINATOR = {
    "linear": pi,          # 180 degrees
    "trigonal_planar": 2 * pi / 3,   # 120 degrees
    "tetrahedral": pi / 2,           # reference-angle branch used in Eq. 16
    "trigonal_bipyramidal": 2 * pi / 5,
    "octahedral": pi / 3,
}


def register_geometry(name: str, max_angle_radians: float) -> None:
    """Register a custom molecular geometry (or override a default one)."""
    _GEOMETRY_DENOMINATOR[name] = float(max_angle_radians)


@dataclass(frozen=True)
class MFV:
    """A single Molecular Fuzzy Value: (membership, non-membership, hesitancy)."""

    m: float
    n: float
    h: float
    # Validation slack, not part of the value: a different tolerance must
    # not make two otherwise identical MFVs unequal or hash differently.
    # NOTE: the sum-to-one check below deliberately keeps its own looser
    # threshold so existing inputs keep validating exactly as before.
    tol: float = field(default=1e-6, repr=False, compare=False)

    def __post_init__(self):
        total = self.m + self.n + self.h
        if abs(total - 1.0) > 1e-3:
            raise ValueError(
                f"MFV components must sum to 1 (got m+n+h={total:.4f})"
            )
        for name, val in (("m", self.m), ("n", self.n), ("h", self.h)):
            if val < -self.tol:
                raise ValueError(f"MFV component {name}={val} must be >= 0")

    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.m, self.n, self.h)

    def scale(self, weight: float) -> "WeightedTriple":
        """Weighted element b = (w.m, w.n, w.h) -- Eq. 22.

        A weighted MFV no longer sums to 1 by construction (that is
        expected: weighted elements feed vector-angle computations, not
        further MFV algebra), so this returns the unvalidated
        :class:`WeightedTriple` twin instead of a strict ``MFV``.
        """
        return WeightedTriple(weight * self.m, weight * self.n, weight * self.h)

    @staticmethod
    def from_angle(alpha: float, max_alpha: float) -> "MFV":
        """Build an MFV from a normalized angle via Eq. 1-3.

        m = 1 - alpha/max_alpha   (Eq. 1)
        n = alpha/max_alpha       (Eq. 2)
        h = 1 - m - n             (Eq. 3)
        """
        ratio = 0.0 if max_alpha == 0 else alpha / max_alpha
        ratio = min(max(ratio, 0.0), 1.0)
        m = 1.0 - ratio
        n = ratio
        h = max(0.0, 1.0 - m - n)
        return MFV(m, n, h)

    @staticmethod
    def average(values: Sequence["MFV"]) -> "MFV":
        """Componentwise mean of several MFVs -- Eq. 13."""
        arr = np.array([v.as_tuple() for v in values], dtype=float)
        mean = arr.mean(axis=0)
        mean = mean / mean.sum()  # guard against rounding drift
        return MFV(*mean)


@dataclass(frozen=True)
class WeightedTriple:
    """An (m, n, h) triple that need not sum to 1 -- e.g. the weighted
    element b = w.(m, n, h) of Eq. 22. Kept distinct from ``MFV`` so the
    latter's sum-to-one invariant stays meaningful everywhere else."""

    m: float
    n: float
    h: float

    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.m, self.n, self.h)


FuzzyVector = List  # sequence of objects exposing .as_tuple() -> (m, n, h)


def _vector_to_array(vec: Iterable) -> np.ndarray:
    return np.array([item.as_tuple() for item in vec], dtype=float).ravel()


def angle_between(vec_a: Iterable[MFV], vec_b: Iterable[MFV]) -> float:
    """Cosine-similarity angle between two fuzzy vectors (Eq. 15 / Eq. 24).

    Each fuzzy vector is the concatenation of (m, n, h) triples of its MFV
    elements; the angle is the arccos of their cosine similarity.
    """
    a = _vector_to_array(vec_a)
    b = _vector_to_array(vec_b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    cos_theta = float(np.dot(a, b) / denom)
    cos_theta = min(1.0, max(-1.0, cos_theta))
    return acos(cos_theta)


def normalize_angle(alpha: float, geometry: str = "linear") -> float:
    """Piecewise geometry-dependent angle normalization (Eq. 16)."""
    if geometry not in _GEOMETRY_DENOMINATOR:
        raise ValueError(
            f"Unknown molecular geometry '{geometry}'. "
            f"Available: {sorted(_GEOMETRY_DENOMINATOR)}"
        )
    denom = _GEOMETRY_DENOMINATOR[geometry]
    return 0.0 if denom == 0 else alpha / denom


def reciprocal_matrix(angle_matrix: np.ndarray) -> np.ndarray:
    """Elementwise reciprocal with zero-safe diagonal (Eq. 17)."""
    with np.errstate(divide="ignore"):
        recip = np.where(angle_matrix > 0, 1.0 / angle_matrix, 0.0)
    np.fill_diagonal(recip, 0.0)
    return recip


def column_normalize(matrix: np.ndarray) -> np.ndarray:
    """Divide every entry by the sum of its column (NR-matrix step)."""
    col_sums = matrix.sum(axis=0)
    col_sums = np.where(col_sums == 0, 1.0, col_sums)
    return matrix / col_sums


def build_angle_matrix(rows: Sequence[FuzzyVector], geometry: str) -> np.ndarray:
    """Full pairwise, geometry-normalized angle matrix for a set of fuzzy
    vectors (one per criterion row or per alternative row)."""
    k = len(rows)
    raw = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            raw[i, j] = angle_between(rows[i], rows[j])
    normalized = np.vectorize(lambda a: normalize_angle(a, geometry))(raw)
    np.fill_diagonal(normalized, 0.0)
    return normalized


def nr_matrix(rows: Sequence[FuzzyVector], geometry: str = "linear") -> np.ndarray:
    """Full NR-matrix pipeline: angle -> normalize -> reciprocal -> column
    normalize (Eq. 15-17)."""
    angles = build_angle_matrix(rows, geometry)
    recip = reciprocal_matrix(angles)
    return column_normalize(recip)
