"""Test the filter operator."""
import starry
import pytest
import numpy as np
np.random.seed(43)


@pytest.fixture(
    scope="class",
    params=[
        (5, 0, False),
        (5, 2, False),
        (5, 0, True),
        (5, 2, True)
    ],
)
def map(request):
    ydeg, udeg, reflected = request.param
    map = starry.Map(ydeg=ydeg, udeg=udeg, fdeg=2, reflected=reflected)
    map[1:, :] = 0.05 * np.random.randn(map.Ny - 1)
    if udeg > 0:
        map[1:] = 0.1 * np.random.randn(map.udeg)
    map.filter[1:, :] = np.random.randn(map.Nf - 1)
    return map


class TestFilter:
    """Test the filter operator."""

    def test_getter_setter(self, map):
        pass # TODO

    def test_phase_curve_static_single_wav(self, map):
        theta = np.linspace(0, 180, 100)

        # Compute the flux analytically
        flux = np.array(map.flux(theta=theta))
        flux /= np.nanmedian(flux)

        # Compute it numerically
        flux_num = np.nansum(map.render(res=150, theta=theta), axis=(1, 2))
        flux_num /= np.nanmedian(flux_num)

        # Compare
        assert np.allclose(flux_num, flux, atol=1e-2, rtol=1e-2)


def test_filter_flux_exact():
    # Occultation flux for a zero-degree map with the filter `Y_{1, 1}`
    map = starry.Map(0, fdeg=1)
    map.filter[1, 1] = 1
    f1 = np.array(map.flux(xo=0.1, ro=0.1))[0]

    # The operation above corresponds to the flux from a map
    # with just the `Y_{1, 1}` term set, normalized by the
    # amplitude of the `Y_{0, 0}` term.
    map = starry.Map(1)
    fbase = np.array(map.flux(xo=0.1, ro=0.1))[0]
    map[1, 1] = 1 / np.pi
    f2 = np.array(map.flux(xo=0.1, ro=0.1))[0] - fbase

    assert np.allclose(f1, f2)