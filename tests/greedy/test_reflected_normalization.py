"""
Starry was originally designed for maps in thermal light.
Maps in reflected light are quirky in how things are normalized,
so we need to make sure albedos, intensities, and fluxes are
all self-consistent in the code. There are some fudges we had
to make under the hood to ensure all of these tests pass.

"""

import numpy as np
import starry


starry.config.lazy = False
map = starry.Map(1, reflected=True)


def test_amp():
    assert map.amp == 1.0


def test_flux():
    assert np.allclose(map.flux(zs=1, xs=0, ys=0), 2.0 / 3.0)


def test_design_matrix():
    assert np.allclose(
        map.amp * map.design_matrix(zs=1, xs=0, ys=0)[0, 0], 2.0 / 3.0
    )


def test_intensity():
    assert np.allclose(
        map.intensity(lat=0, lon=0, zs=1, xs=0, ys=0), 1.0 / np.pi
    )


def test_albedo():
    assert np.allclose(
        map.amp
        * map.intensity(lat=0, lon=0, zs=1, xs=0, ys=0, illuminate=False),
        1.0,
    )


def test_albedo_design_matrix():
    assert np.allclose(
        map.amp * map.intensity_design_matrix(lat=0, lon=0)[0, 0], 1.0
    )


def test_render():
    assert np.allclose(np.nanmax(map.render(zs=1, xs=0, ys=0)), 1.0 / np.pi)


def test_render_albedo():
    assert np.allclose(
        np.nanmax(map.render(zs=1, xs=0, ys=0, illuminate=False)), 1.0
    )


def test_numerical_flux(res=300):
    res = 300
    atol = 1e-4
    rtol = 1e-4
    num_flux = np.nansum(map.render(zs=1, xs=0, ys=0, res=res)) * 4 / res ** 2
    assert np.allclose(num_flux, 2.0 / 3.0, atol=atol, rtol=rtol)


def test_solve():
    # Generate a light curve with two points: one at
    # full phase, one at new phase.
    np.random.seed(0)
    ferr = 1e-8
    flux = np.array([2.0 / 3.0, 0])
    flux += ferr * np.random.randn(len(flux))
    map.set_data(flux, C=ferr ** 2)

    # Make the prior very informative for the coeffs
    # but extremely wide for the amplitude (with zero mean)
    mu = np.zeros(4)
    L = np.array([1e1, 1e-12, 1e-12, 1e-12])
    map.set_prior(mu=mu, L=L)

    # Check that we recover the correct amplitude
    x, _ = map.solve(zs=[1.0, -1.0])
    assert np.allclose(x[0], 1.0)
    assert np.allclose(map.amp, 1.0)

    # Check that a map draw gives us the correct amplitude
    map.draw()
    assert np.allclose(map.amp, 1.0)


def test_one_over_r_squared(n_tests=10, plot=False):
    """Test that the flux decreases as 1/r^2."""
    flux0 = map.flux()
    zs = np.linspace(1, 10, 100)
    flux = map.flux(xs=0, ys=0, zs=zs)
    assert np.allclose(flux, flux0 / zs ** 2)


def test_sys_flux():
    """Test the normalization of the flux."""
    # Instantiate a system. Planet has radius `r` and is at
    # distance `d` from a point illumination source.
    d = 10
    r = 2
    planet = starry.Secondary(starry.Map(reflected=True), a=d, r=r)
    star = starry.Primary(starry.Map(), r=0)
    sys = starry.System(star, planet)

    # Get the star & planet flux when it's at full phase
    t_full = 0.5 * sys._get_periods()[0]
    f_star, f_planet = sys.flux(t=t_full, total=False)

    # Star should have unit flux
    assert np.allclose(f_star, 1.0)

    # Planet should have flux equal to (2 / 3) r^2 / d^2
    f_expected = (2.0 / 3.0) * r ** 2 / d ** 2
    assert np.allclose(f_planet, f_expected)
