import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from molfuzzy.aco import aco_rank
from molfuzzy.cognitive_map import aggregate_matrices, cognitive_map_weights
from molfuzzy.lom import compute_reliability_dimensions, lom_expert_weights
from molfuzzy.mfv import MFV
from molfuzzy.qlearning import balance_experts


def _toy_number_matrices():
    m1 = np.array([[0, 5, 4], [4, 0, 3], [3, 4, 0]], dtype=float)
    m2 = np.array([[0, 4, 4], [3, 0, 3], [4, 3, 0]], dtype=float)
    m3 = np.array([[0, 2, 5], [5, 0, 1], [1, 5, 0]], dtype=float)
    return [m1, m2, m3]


def test_lom_weights_sum_to_one_and_are_nonnegative():
    dims = compute_reliability_dimensions(_toy_number_matrices())
    result = lom_expert_weights(dims)
    assert result["weights"].sum() == pytest.approx(1.0, abs=1e-6)
    assert np.all(result["weights"] >= -1e-9)
    assert 0 <= result["benchmark"] < 3


def test_lom_rejects_bad_beta():
    dims = compute_reliability_dimensions(_toy_number_matrices())
    with pytest.raises(ValueError):
        lom_expert_weights(dims, beta=(0.5, 0.5, 0.5))


def _toy_mfv_matrix(seed_offset=0.0):
    m_val = min(0.95, 0.8 + seed_offset)
    n_val = max(0.02, 0.17 - seed_offset)
    h_val = max(0.0, 1.0 - m_val - n_val)
    hi = MFV(m_val, n_val, h_val)
    lo = MFV(0.6, 0.3, 0.1)
    return [
        [MFV(0, 0, 1), hi, lo],
        [hi, MFV(0, 0, 1), hi],
        [lo, hi, MFV(0, 0, 1)],
    ]


def _mfv_matrix_to_array(matrix):
    """(n, n, 3) numeric view of a matrix of MFVs."""
    return np.array([[cell.as_tuple() for cell in row] for row in matrix], dtype=float)


def test_balance_experts_leaves_benchmark_untouched():
    matrices = [_toy_mfv_matrix(0.0), _toy_mfv_matrix(0.1), _toy_mfv_matrix(-0.2)]
    result = balance_experts(matrices, weights=[0.4, 0.35, 0.25], benchmark=0, learning_rate=0.3)
    assert set(result.balanced.keys()) == {0, 1, 2}
    for i in range(3):
        for j in range(3):
            if i != j:
                orig = matrices[0][i][j].as_tuple()
                bal = result.balanced[0][i][j].as_tuple()
                assert orig == pytest.approx(bal)


def test_balance_experts_moves_toward_benchmark():
    """Every non-benchmark expert must end up strictly closer to the benchmark.

    This asserts actual movement, not merely that the benchmark survived
    (which the previous version of this test only checked).
    """
    matrices = [_toy_mfv_matrix(0.0), _toy_mfv_matrix(0.1), _toy_mfv_matrix(-0.2)]
    result = balance_experts(matrices, weights=[0.4, 0.35, 0.25], benchmark=0, learning_rate=0.3)
    before = [_mfv_matrix_to_array(m) for m in matrices]
    after = [_mfv_matrix_to_array(result.balanced[k]) for k in range(3)]
    for k in (1, 2):
        d_before = float(np.max(np.abs(before[k] - before[0])))
        d_after = float(np.max(np.abs(after[k] - after[0])))
        assert d_after < d_before, (
            f"expert {k} did not move toward the benchmark "
            f"(max elementwise gap {d_before:.6f} -> {d_after:.6f})"
        )


def test_cognitive_map_weights_sum_to_one():
    matrices = [_toy_mfv_matrix(0.0), _toy_mfv_matrix(0.05)]
    aggregated = aggregate_matrices(matrices)
    result = cognitive_map_weights(aggregated, geometry="linear")
    assert result.weights.sum() == pytest.approx(1.0)
    assert np.all(result.weights > 0)


def test_aco_rank_returns_full_permutation():
    decision_matrix = [
        [MFV(0.8, 0.15, 0.05), MFV(0.6, 0.3, 0.1)],
        [MFV(0.9, 0.05, 0.05), MFV(0.7, 0.2, 0.1)],
        [MFV(0.5, 0.4, 0.1), MFV(0.95, 0.05, 0.0)],
    ]
    labels = ["Alt1", "Alt2", "Alt3"]
    result = aco_rank(decision_matrix, weights=[0.5, 0.5], labels=labels, seed=1)
    assert sorted(result.ranking) == [0, 1, 2]
    assert len(result.ranking_labels) == 3
    assert result.fitness > 0


def test_aco_rank_is_seed_reproducible():
    decision_matrix = [
        [MFV(0.8, 0.15, 0.05), MFV(0.6, 0.3, 0.1)],
        [MFV(0.9, 0.05, 0.05), MFV(0.7, 0.2, 0.1)],
        [MFV(0.5, 0.4, 0.1), MFV(0.95, 0.05, 0.0)],
    ]
    labels = ["Alt1", "Alt2", "Alt3"]
    r1 = aco_rank(decision_matrix, weights=[0.5, 0.5], labels=labels, seed=7)
    r2 = aco_rank(decision_matrix, weights=[0.5, 0.5], labels=labels, seed=7)
    assert r1.ranking == r2.ranking
    assert r1.fitness == pytest.approx(r2.fitness)
