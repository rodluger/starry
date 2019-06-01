# -*- coding: utf-8 -*-
"""
Test against numerical integration.

"""
import theano
import starry
import numpy as np
res = 1000
map_tmp = starry.Map(1)
compute_ortho_grid = theano.function([], map_tmp.ops.compute_ortho_grid(res))


def flux(map, theta=0, xo=0, yo=0, ro=0):
    x, y, z = compute_ortho_grid()
    occulted = (xo - x) ** 2 + (yo - y) ** 2 < ro ** 2
    occulted = occulted.reshape(res, res)
    image = map.render(theta=theta, res=res)[:, :, 0]
    image[occulted] = 0.0
    image *= 4.0 / (res ** 2)
    return np.nansum(image)


def test_flux(n_tests=10):
    map = starry.Map(2, lazy=False)

    np.random.seed(12)
    for i in range(n_tests):
        map[1:, :] = np.random.randn(len(map[1:, :]))
        theta = np.random.random() * 360
        xo = np.random.randn()
        yo = np.random.randn()
        ro = 0.5 * np.random.random()
        kwargs = dict(theta=theta, xo=xo, yo=yo, ro=ro)
        assert np.allclose(map.flux(**kwargs), 
                           flux(map, **kwargs),
                           rtol=1e-3,
                           atol=1e-3)


if __name__ == "__main__":
    test_flux()