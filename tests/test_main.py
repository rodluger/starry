"""Test the starry code."""
from starry import Map
import numpy as np


def test_main():
    """Main routines."""
    # Check the Map class
    m = Map(1)
    m.set_coeff(1, 0, 1)
    assert np.allclose(m.y, np.array([1, 0, 1, 0]))

    # Rotations and evaluations
    m.rotate([1, 0, 0], -np.pi / 2)
    assert np.allclose(m.y, np.array([1, 1, 0, 0]))
    assert np.allclose(m.p, np.array([np.sqrt(1 / (4 * np.pi)), 0, 0,
                                      np.sqrt(3 / (4 * np.pi))]))

    m.rotate([0, 0, 1], -np.pi / 2)
    assert np.allclose(m.y, np.array([1, 0, 0, 1]))
    assert np.allclose(m.p, np.array([np.sqrt(1 / (4 * np.pi)),
                                      np.sqrt(3 / (4 * np.pi)), 0, 0]))

    m.rotate([0, 1, 0], -np.pi / 2)
    assert np.allclose(m.y, np.array([1, 0, 1, 0]))
    assert np.allclose(m.p, np.array([np.sqrt(1 / (4 * np.pi)), 0,
                                      np.sqrt(3 / (4 * np.pi)), 0]))


if __name__ == "__main__":
    test_main()
