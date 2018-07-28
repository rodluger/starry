"""Test the rotation matrices."""
from starry2 import SurfaceMap
import numpy as np


def test_rotations():
    """Test some elementary rotations."""
    # Instantiate
    m = SurfaceMap(1)
    m.set_coeff(1, 0, 1)
    assert np.allclose(m.y, np.array([0, 0, 1, 0]))

    # Elemental rotations
    m.axis = [1, 0, 0]
    m.rotate(-90 * np.pi / 180)
    assert np.allclose(m.y, np.array([0, 1, 0, 0]))
    assert np.allclose(m.p, np.array([0, 0, 0,
                                      np.sqrt(3 / (4 * np.pi))]))

    m.axis = [0, 0, 1]
    m.rotate(-90 * np.pi / 180)
    assert np.allclose(m.y, np.array([0, 0, 0, 1]))
    assert np.allclose(m.p, np.array([0,
                                      np.sqrt(3 / (4 * np.pi)), 0, 0]))
    m.axis = [0, 1, 0]
    m.rotate(-90 * np.pi / 180)
    assert np.allclose(m.y, np.array([0, 0, 1, 0]))
    assert np.allclose(m.p, np.array([0, 0,
                                      np.sqrt(3 / (4 * np.pi)), 0]))

    # A more complex rotation
    m = SurfaceMap(5)
    m[:] = 1
    m.axis = [1, 2, 3]
    m.rotate(60 * np.pi / 180)
    benchmark = np.array([1.,  1.39148148,  0.91140212,  0.48283069,
                          1.46560344, 0.68477955,  0.30300625,  1.33817773,
                          -0.70749636, 0.66533701, 1.5250326,  0.09725931,
                          -0.13909678,  1.06812054, 0.81540106, -1.54823596,
                          -0.50475248,  1.90009363, 0.68002942, -0.10159448,
                          -0.48406777,  0.59834505, 1.22007458, -0.27302899,
                          -1.58323797, -1.37266583, 1.44638769,  1.36239055,
                          0.22257365, -0.24387785, -0.62003044,  0.03888137,
                          1.05768142,  0.87317586, -1.46092763, -0.81070502])
    assert np.allclose(m.y, benchmark)


if __name__ == "__main__":
    test_rotations()
