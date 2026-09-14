import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from molfuzzy.mfv import MFV, angle_between, nr_matrix, normalize_angle


def test_mfv_valid_triple():
    v = MFV(0.6, 0.3, 0.1)
    assert v.as_tuple() == (0.6, 0.3, 0.1)


def test_mfv_rejects_bad_sum():
    with pytest.raises(ValueError):
        MFV(0.6, 0.3, 0.3)


def test_mfv_rejects_negative():
    with pytest.raises(ValueError):
        MFV(-0.1, 0.6, 0.5)


def test_scale_produces_weighted_triple():
    v = MFV(0.8, 0.15, 0.05)
    scaled = v.scale(0.5)
    assert scaled.as_tuple() == pytest.approx((0.4, 0.075, 0.025))


def test_average_sums_to_one():
    a = MFV(0.95, 0.05, 0.0)
    b = MFV(0.20, 0.70, 0.10)
    avg = MFV.average([a, b])
    assert sum(avg.as_tuple()) == pytest.approx(1.0)


def test_angle_between_identical_vectors_is_zero():
    a = MFV(0.8, 0.15, 0.05)
    theta = angle_between([a], [a])
    assert theta == pytest.approx(0.0, abs=1e-9)


def test_angle_between_orthogonal_like_vectors_is_positive():
    a = MFV(1.0, 0.0, 0.0)
    b = MFV(0.0, 1.0, 0.0)
    theta = angle_between([a], [b])
    assert theta > 0


def test_normalize_angle_known_geometries():
    assert normalize_angle(math.pi, "linear") == pytest.approx(1.0)
    assert normalize_angle(2 * math.pi / 3, "trigonal_planar") == pytest.approx(1.0)
    with pytest.raises(ValueError):
        normalize_angle(1.0, "not_a_geometry")


def test_nr_matrix_shape_and_zero_diagonal():
    rows = [
        [MFV(0.8, 0.15, 0.05), MFV(0.6, 0.3, 0.1)],
        [MFV(0.6, 0.3, 0.1), MFV(0.9, 0.05, 0.05)],
        [MFV(0.4, 0.5, 0.1), MFV(0.95, 0.05, 0.0)],
    ]
    nr = nr_matrix(rows, "linear")
    assert nr.shape == (3, 3)
    assert np.allclose(np.diag(nr), 0.0)
    assert np.all(nr >= 0)
