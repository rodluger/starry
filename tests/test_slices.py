"""Test slice indexing of the map."""
import starry2
import numpy as np


def test_scalar():
    """Test slice indexing for scalar maps."""
    # Do the default and the multi maps
    for map in starry2.Map(5), starry2.Map(5, multi=True):
        # No slice
        map[0, 0] = 1
        assert map.y[0] == 1
        map.reset()

        # l slice
        map[:, 0] = 1
        assert np.allclose(map[:, 0], 1)
        assert map[0, 0] == map[1, 0] == map[2, 0] == 1
        map.reset()

        # m slice
        map[2, :] = 1
        assert np.allclose(map[2, :], 1)
        assert map[2, -2] == map[2, -1] == map[2, 0] == \
            map[2, 1] == map[2, 2] == 1
        map.reset()

        # Both slices
        map[:, :] = 1
        assert(np.allclose(map[:, :], 1))
        assert np.allclose(map.y, 1)

        # Vector assignment
        map[:, 0] = [0, 1, 2, 3, 4, 5]
        assert np.allclose(map[:, 0], [0, 1, 2, 3, 4, 5])
        assert (map[0, 0] == 0) and (map[1, 0] == 1) and (map[2, 0] == 2) \
            and (map[3, 0] == 3) and (map[4, 0] == 4) and (map[5, 0] == 5)
        map.reset()


def test_spectral():
    """Test slice indexing for spectral maps."""

    # DEBUG! Spectral disabled for now.
    return

    # Let's do two wavelength bins
    map = starry2.Map(5, 2)

    # No slice
    map[0, 0] = [1, 2]
    assert np.allclose(map.y[0], [1, 2])
    map.reset()

    # l slice
    map[:, 0] = [1, 2]
    assert np.allclose(map[:, 0], [1, 2])
    assert np.allclose(map[0, 0], [1, 2]) and \
        np.allclose(map[1, 0], [1, 2]) and \
        np.allclose(map[2, 0], [1, 2])
    map.reset()

    # m slice
    map[2, :] = [1, 2]
    assert np.allclose(map[2, :], [1, 2])
    assert np.allclose(map[2, -2], [1, 2]) and \
        np.allclose(map[2, -1], [1, 2]) and \
        np.allclose(map[2, 0], [1, 2]) and \
        np.allclose(map[2, 1], [1, 2]) and \
        np.allclose(map[2, 2], [1, 2])
    map.reset()

    # Both slices
    map[:, :] = [1, 2]
    assert(np.allclose(map[:, :], [1, 2]))
    assert np.allclose(map.y, [1, 2])

    # Vector assignment
    map[:, 0] = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
    assert np.allclose(map[:, 0], [[0, 1], [1, 2], [2, 3],
                                   [3, 4], [4, 5], [5, 6]])
    assert (np.allclose(map[0, 0], [0, 1])) and \
        (np.allclose(map[1, 0], [1, 2])) and \
        (np.allclose(map[2, 0], [2, 3])) and \
        (np.allclose(map[3, 0], [3, 4])) and \
        (np.allclose(map[4, 0], [4, 5])) and \
        (np.allclose(map[5, 0], [5, 6]))
    map.reset()


if __name__ == "__main__":
    test_scalar()
    test_spectral()
