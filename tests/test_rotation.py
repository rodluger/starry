"""Test the rotation matrices."""
import starry2
import numpy as np
norm = 0.5 * np.sqrt(np.pi)


def run(Map):
    """Apply some elementary rotations."""
    # Instantiate
    map = Map(1)
    map[0, 0] = 0
    map[1, 0] = norm
    assert np.allclose(map.y, np.array([0, 0, norm, 0]))

    # Elemental rotations
    map.axis = [1, 0, 0]
    map.rotate(-90)
    assert np.allclose(map.y, np.array([0, norm, 0, 0]))
    assert np.allclose(map.p, np.array([0, 0, 0,
                                        np.sqrt(3 / (4 * np.pi))]))

    map.axis = [0, 0, 1]
    map.rotate(-90)
    assert np.allclose(map.y, np.array([0, 0, 0, norm]))
    assert np.allclose(map.p, np.array([0,
                                        np.sqrt(3 / (4 * np.pi)), 0, 0]))
    map.axis = [0, 1, 0]
    map.rotate(-90)
    assert np.allclose(map.y, np.array([0, 0, norm, 0]))
    assert np.allclose(map.p, np.array([0, 0,
                                        np.sqrt(3 / (4 * np.pi)), 0]))

    # A more complex rotation
    map = Map(5)
    map[:, :] = 1
    map.axis = [1, 2, 3]
    map.rotate(60)
    benchmark = np.array([1.,  1.39148148,  0.91140212,  0.48283069,
                          1.46560344, 0.68477955,  0.30300625,  1.33817773,
                          -0.70749636, 0.66533701, 1.5250326,  0.09725931,
                          -0.13909678,  1.06812054, 0.81540106, -1.54823596,
                          -0.50475248,  1.90009363, 0.68002942, -0.10159448,
                          -0.48406777,  0.59834505, 1.22007458, -0.27302899,
                          -1.58323797, -1.37266583, 1.44638769,  1.36239055,
                          0.22257365, -0.24387785, -0.62003044,  0.03888137,
                          1.05768142,  0.87317586, -1.46092763, -0.81070502])
    assert np.allclose(map.y, benchmark)

    # Test rotation caching
    map = Map(2)
    map[0, 0] = 0
    map[1, 0] = norm
    map.axis = [1, 2, 3]
    assert np.allclose(map(theta=[30, 30, 30]), 0.46522382467359763)
    map.axis = [3, 2, 1]
    assert np.allclose(map(theta=[30, 30, 30]), 0.42781792510668176)
    map.rotate(30)
    assert np.allclose(map(theta=[30, 30, 30]), 0.2617513456622787)


def test_rotation_double():
    """Test some elementary rotations [double]."""
    return run(starry2.Map)


def test_rotation_multi():
    """Test some elementary rotations [multi]."""
    return run(starry2.multi.Map)


if __name__ == "__main__":
    test_rotation_double()
    test_rotation_multi()
