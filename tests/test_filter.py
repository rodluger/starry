"""Test the filter operator."""
import starry
import pytest
import numpy as np
np.random.seed(43)


@pytest.fixture(
    scope="class",
    params=[
        (5, 0),
        (5, 2),
        (5, 0),
        (5, 2)
    ],
)
def map(request):
    ydeg, udeg = request.param
    map = starry.Map(ydeg=ydeg, udeg=udeg, fdeg=2)
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