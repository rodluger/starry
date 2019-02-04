"""Test the stability of the limb darkening calculations near singular points."""
import starry2
import numpy as np
import pytest

# We'll run all tests on tenth degree maps
map = starry2.Map(10)
map[0, 0] = 1
map[:] = 1
map_multi = starry2.Map(10, multi=True)
map_multi[0, 0] = 1
map_multi[:] = 1


def test_b_near_zero(ftol=1e-11, gtol=1e-11):
    b = 10.0 ** np.linspace(-18, -6, 100)
    b[0] = 0
    f = map.flux(xo=b, yo=0.0, ro=0.1)
    f_multi = map_multi.flux(xo=b, yo=0.0, ro=0.1)
    assert(np.max(np.abs(f - f_multi)) < ftol)
    _, g = map.flux(xo=b, yo=0.0, ro=0.1, gradient=True)
    _, g_multi = map_multi.flux(xo=b, yo=0.0, ro=0.1, gradient=True)
    for key in g.keys():
        assert(np.max(np.abs(g[key] - g_multi[key])) < gtol)


def test_b_near_one_plus_r(ftol=1e-11, gtol=1e-11):
    ro = 0.1
    b = 1 + ro - 10.0 ** np.linspace(-18, -6, 100)
    b[0] = 1 + ro
    f = map.flux(xo=b, yo=0.0, ro=ro)
    f_multi = map_multi.flux(xo=b, yo=0.0, ro=ro)
    assert(np.max(np.abs(f - f_multi)) < ftol)
    _, g = map.flux(xo=b, yo=0.0, ro=ro, gradient=True)
    _, g_multi = map_multi.flux(xo=b, yo=0.0, ro=ro, gradient=True)
    for key in g.keys():
        assert(np.max(np.abs(g[key] - g_multi[key])) < gtol)


def test_b_near_one_minus_r(ftol=1e-11, gtol=1e-11):
    ro = 0.1
    b = 1 - ro - 10.0 ** np.linspace(-18, -6, 100)
    b[0] = 1 - ro
    f = map.flux(xo=b, yo=0.0, ro=ro)
    f_multi = map_multi.flux(xo=b, yo=0.0, ro=ro)
    assert(np.max(np.abs(f - f_multi)) < ftol)
    _, g = map.flux(xo=b, yo=0.0, ro=ro, gradient=True)
    _, g_multi = map_multi.flux(xo=b, yo=0.0, ro=ro, gradient=True)
    for key in g.keys():
        assert(np.max(np.abs(g[key] - g_multi[key])) < gtol)


def test_b_equals_r_equals_half(ftol=1e-11, gtol=1e-5):
    # TODO: The gradient precision is pretty 
    # poor in this case. This is a *crazy* unlikely
    # edge case, though.
    ro = 0.5
    b = 0.5 - 10.0 ** np.linspace(-18, -6, 100)
    b[0] = 0.5
    f = map.flux(xo=b, yo=0.0, ro=ro)
    f_multi = map_multi.flux(xo=b, yo=0.0, ro=ro)
    assert(np.max(np.abs(f - f_multi)) < ftol)
    _, g = map.flux(xo=b, yo=0.0, ro=ro, gradient=True)
    _, g_multi = map_multi.flux(xo=b, yo=0.0, ro=ro, gradient=True)
    for key in g.keys():
        assert(np.max(np.abs(g[key] - g_multi[key])) < gtol)


def test_r_near_one_b_is_zero(ftol=1e-12, gtol=1e-11):
    ro = 0.5 - 10.0 ** np.linspace(-18, -6, 100)
    ro[0] = 0.5
    b = 0.0
    f = map.flux(xo=b, yo=0.0, ro=ro)
    f_multi = map_multi.flux(xo=b, yo=0.0, ro=ro)
    assert(np.max(np.abs(f - f_multi)) < ftol)
    _, g = map.flux(xo=b, yo=0.0, ro=ro, gradient=True)
    _, g_multi = map_multi.flux(xo=b, yo=0.0, ro=ro, gradient=True)
    for key in g.keys():
        assert(np.max(np.abs(g[key] - g_multi[key])) < gtol)


def test_r_large(ftol=1e-6, gtol=1e-5):
    # TODO: For r = 100, we currently only get ~ppm precision
    # on the flux and its gradient for l = 10. This is kind of a 
    # bummer -- let's think of ways to improve this.
    ro = 100
    b = np.linspace(ro - 1, ro + 1, 500)
    f = map.flux(xo=b, yo=0.0, ro=ro)
    f_multi = map_multi.flux(xo=b, yo=0.0, ro=ro)
    assert(np.max(np.abs(f - f_multi)) < ftol)
    _, g = map.flux(xo=b, yo=0.0, ro=ro, gradient=True)
    _, g_multi = map_multi.flux(xo=b, yo=0.0, ro=ro, gradient=True)
    for key in g.keys():
        assert(np.max(np.abs(g[key] - g_multi[key])) < gtol)


def test_r_small(ftol=1e-11, gtol=1e-11):
    b = 0.5
    ro = 10 ** np.linspace(-18, -1, 500)
    f = map.flux(xo=b, yo=0.0, ro=ro)
    f_multi = map_multi.flux(xo=b, yo=0.0, ro=ro)
    assert(np.max(np.abs(f - f_multi)) < ftol)
    _, g = map.flux(xo=b, yo=0.0, ro=ro, gradient=True)
    _, g_multi = map_multi.flux(xo=b, yo=0.0, ro=ro, gradient=True)
    for key in g.keys():
        assert(np.max(np.abs(g[key] - g_multi[key])) < gtol)