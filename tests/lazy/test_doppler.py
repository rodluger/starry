import starry
import numpy as np
import pytest


@pytest.fixture(scope="module", params=[1, 2])
def map(request):
    nc = request.param
    map = starry.DopplerMap(ydeg=10, udeg=2, nt=3, nc=nc, veq=50000)
    map.load(maps=["spot", "earth"][:nc])
    yield map


@pytest.fixture(scope="function")
def random():
    yield np.random.default_rng(0)


def test_ld_indices(map):
    """
    Test limb darkening coeff setting/getting.

    """
    # Set all coeffs
    map[1:] = [0.5, 0.25]
    assert np.array_equal(map._u.eval(), [-1, 0.5, 0.25])
    assert np.array_equal(map[:].eval(), [-1, 0.5, 0.25])

    # Set individual coeff
    map[1] = 0.75
    assert map._u[1].eval() == 0.75
    assert map[1].eval() == 0.75

    # Set individual coeff
    map[2] = 0.10
    assert map._u[2].eval() == 0.10
    assert map[2].eval() == 0.10

    # Attempt to set all coeffs
    with pytest.raises(ValueError):
        map[:] = [0.5, 0.25]


def test_ylm_indices(map, random):
    """
    Test sph harm coeff setting/getting.

    """
    if map.nc == 1:

        # Set all coeffs (1st method)
        y = random.normal(size=map.Ny)
        map[:, :] = y
        assert np.array_equal(map.y.eval(), y)

        # Set all coeffs (2nd method)
        y = random.normal(size=map.Ny)
        map[:, :, :] = y
        assert np.array_equal(map.y.eval(), y)

        # Set all coeffs (3rd method)
        y = random.normal(size=map.Ny)
        map[:, :, 0] = y
        assert np.array_equal(map.y.eval(), y)

        # Set all coeffs (4th method)
        y = random.normal(size=(map.Ny, 1))
        map[:, :, 0] = y
        assert np.array_equal(map.y.eval(), y.reshape(-1))

        # Set one coeff
        y = random.normal()
        l, m = (5, -3)
        map[l, m] = y
        assert map[l, m].eval() == y
        assert map.y[l ** 2 + l + m].eval() == y

        # Set several coeffs (single l, all ms)
        l = 5
        y = random.normal(size=(2 * l + 1))
        map[l, :] = y
        assert np.array_equal(map[l, :].eval().reshape(-1), y)
        assert np.array_equal(
            map.y[l ** 2 : l ** 2 + 2 * l + 1].eval().reshape(-1), y
        )

        # Set several coeffs (l = (4, 5) and m = (3, 4))
        y = random.normal(size=4)
        map[4:6, 3:5] = y
        assert np.array_equal(map[4:6, 3:5].eval().reshape(-1), y)
        assert np.array_equal(
            np.array(
                [
                    map[4, 3].eval(),
                    map[4, 4].eval(),
                    map[5, 3].eval(),
                    map[5, 4].eval(),
                ]
            ).reshape(-1),
            y,
        )

    elif map.nc == 2:

        # Set all coeffs
        y = random.normal(size=(map.Ny, map.nc))
        map[:, :, :] = y
        assert np.array_equal(map.y.eval(), y)

        # Set all coeffs for one component
        y = random.normal(size=map.Ny)
        map[:, :, 0] = y
        assert np.array_equal(map.y[:, 0].eval().reshape(-1), y)
        assert np.array_equal(map[:, :, 0].eval().reshape(-1), y)

        # Set all coeffs for one component (matrix input)
        y = random.normal(size=(map.Ny, 1))
        map[:, :, 0] = y
        assert np.array_equal(map.y[:, 0].eval().reshape(-1), y.reshape(-1))

        # Set one coeff
        y = random.normal()
        l, m, c = (5, -3, 0)
        map[l, m, c] = y
        assert map[l, m, c].eval() == y
        assert map.y[l ** 2 + l + m, c].eval() == y

        # Set several coeffs (single l, all ms, single c)
        l = 5
        c = 0
        y = random.normal(size=(2 * l + 1))
        map[l, :, c] = y
        assert np.array_equal(map[l, :, c].eval().reshape(-1), y)
        assert np.array_equal(
            map.y[l ** 2 : l ** 2 + 2 * l + 1, c].eval().reshape(-1), y
        )

        # Set several coeffs (l = (4, 5) and m = (3, 4), c = 0)
        y = random.normal(size=4)
        map[4:6, 3:5, 0] = y
        assert np.array_equal(map[4:6, 3:5, 0].eval().reshape(-1), y)
        assert np.array_equal(
            np.array(
                [
                    map[4, 3, 0].eval(),
                    map[4, 4, 0].eval(),
                    map[5, 3, 0].eval(),
                    map[5, 4, 0].eval(),
                ]
            ).reshape(-1),
            y,
        )
