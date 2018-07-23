"""Test system stuff."""
import starry
import numpy as np


def assert_allclose(x, y):
    """Shortcut assert with ppm tolerance."""
    return np.testing.assert_allclose(x, y, rtol=1e-6)


def test_system():
    """Run the tests."""
    np.random.seed(123)
    star = starry.Star()
    star.map[1] = 0.40
    star.map[2] = 0.26
    assert np.allclose(star.map.evaluate(x=[0, 1, 0], y=[0, 0, 1]),
                       np.array([0.3866112, 0.13144781, 0.13144781]))
    planet = starry.Planet(lmax=5, r=0.1, L=5e-3, porb=1,
                           prot=1, a=30, Omega=30, ecc=0.3, w=30)
    for l in range(1, planet.map.lmax + 1):
        for m in range(-l, l + 1):
            planet.map[l, m] = 0.01 * np.random.randn()
    system = starry.System([star, planet])
    time = np.linspace(-0.25, 3.25, 10000)
    system.compute(time)
    assert_allclose([np.min(system.flux), np.max(system.flux)],
                    [0.992787453371187, 1.0051091461483344])
    assert_allclose([np.min(star.flux), np.max(star.flux)],
                    [0.9878665127810682, 1.0])
    assert_allclose([np.min(planet.flux), np.max(planet.flux)],
                    [0.0, 0.00510914618136758])
    assert np.argmin(planet.flux) == 2390
    assert_allclose(planet.x[500:505], np.array(
        [-0.30964744, -0.24076855, -0.17188729, -0.10300434, -0.03412038]))
    assert_allclose(planet.y[500:505], np.array(
        [-0.17877503, -0.13900779, -0.09923917, -0.05946958, -0.01969941]))
    assert_allclose(planet.z[500:505], np.array(
        [23.65600332, 23.67490184, 23.6935674,  23.71200035, 23.73020103]))


if __name__ == "__main__":
    test_system()
