# -*- coding: utf-8 -*-
"""
Test against numerical integration.

"""
import theano
import starry
import numpy as np

res = 1000
map_tmp = starry.Map(1)


def flux(map, xs=0, ys=0, zs=0, xo=0, yo=0, zo=1, ro=0, **kwargs):
    (lat, lon), (x, y, z) = map_tmp.ops.compute_ortho_grid(res)
    occulted = (xo - x) ** 2 + (yo - y) ** 2 < ro ** 2
    occulted = occulted.reshape(res, res)
    if map._ops_class_ == starry._core.OpsReflected:
        image = map.render(res=res, xs=xs, ys=ys, zs=zs, **kwargs)
    else:
        image = map.render(res=res, **kwargs)
    image[occulted] = 0.0
    image *= 4.0 / (res ** 2)
    return np.nansum(image)


def test_flux(n_tests=10):
    map = starry.Map(2)
    np.random.seed(12)
    for i in range(n_tests):
        map[1:, :] = np.random.randn(len(map[1:, :]))
        theta = np.random.random() * 360
        xo = np.random.randn()
        yo = np.random.randn()
        ro = 0.5 * np.random.random()
        kwargs = dict(theta=theta, xo=xo, yo=yo, ro=ro)
        assert np.allclose(
            map.flux(**kwargs), flux(map, **kwargs), rtol=1e-3, atol=1e-3
        )


def test_flux_reflected(n_tests=10):
    map = starry.Map(2, reflected=True)
    np.random.seed(13)
    for i in range(n_tests):
        map[1:, :] = 0.1 * np.random.randn(len(map[1:, :]))
        theta = np.random.random() * 360
        source = np.random.randn(3)
        kwargs = dict(
            theta=theta, xs=source[0], ys=source[1], zs=source[2], ro=0
        )
        assert np.allclose(
            map.flux(**kwargs), flux(map, **kwargs), rtol=1e-3, atol=1e-3
        )


if __name__ == "__main__":
    test_flux_reflected()
