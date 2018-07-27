"""Test the rotation matrices."""
from starry2 import SurfaceMap
import numpy as np


def test_rotations():
    """Test some elementary rotations."""
    # Instantiate
    m = SurfaceMap(1)
    m.set_coeff(1, 0, 1)
    assert np.allclose(m.y, np.array([0, 0, 1, 0]))

    # Rotations and evaluations
    m.axis = [1, 0, 0]
    m.rotate(-90)
    assert np.allclose(m.y, np.array([0, 1, 0, 0]))
    assert np.allclose(m.p, np.array([0, 0, 0,
                                      np.sqrt(3 / (4 * np.pi))]))

    m.axis = [0, 0, 1]
    m.rotate(-90)
    assert np.allclose(m.y, np.array([0, 0, 0, 1]))
    assert np.allclose(m.p, np.array([0,
                                      np.sqrt(3 / (4 * np.pi)), 0, 0]))
    m.axis = [0, 1, 0]
    m.rotate(-90)
    assert np.allclose(m.y, np.array([0, 0, 1, 0]))
    assert np.allclose(m.p, np.array([0, 0,
                                      np.sqrt(3 / (4 * np.pi)), 0]))


if __name__ == "__main__":
    test_rotations()
