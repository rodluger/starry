"""Test the rotation matrices."""
from starry import Map
import numpy as np


def test_rotations():
    """Test some elementary rotations."""
    # Instantiate
    m = Map(1)
    m.set_coeff(1, 0, 1)
    assert np.allclose(m.y, np.array([0, 0, 1, 0]))

    # Rotations and evaluations
    m.rotate([1, 0, 0], -np.pi / 2)
    assert np.allclose(m.y, np.array([0, 1, 0, 0]))
    assert np.allclose(m.p, np.array([0, 0, 0,
                                      np.sqrt(3 / (4 * np.pi))]))

    m.rotate([0, 0, 1], -np.pi / 2)
    assert np.allclose(m.y, np.array([0, 0, 0, 1]))
    assert np.allclose(m.p, np.array([0,
                                      np.sqrt(3 / (4 * np.pi)), 0, 0]))

    m.rotate([0, 1, 0], -np.pi / 2)
    assert np.allclose(m.y, np.array([0, 0, 1, 0]))
    assert np.allclose(m.p, np.array([0, 0,
                                      np.sqrt(3 / (4 * np.pi)), 0]))


if __name__ == "__main__":
    test_rotations()
