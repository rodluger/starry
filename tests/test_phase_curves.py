"""Test phase curves against numerically computed fluxes."""
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
    map = starry.Map(ydeg=ydeg, udeg=udeg, reflected=reflected)
    map[1:, :] = 0.05 * np.random.randn(map.Ny - 1)
    if udeg > 0:
        map[1:] = 0.5
    return map


class TestPhaseCurves:
    """Test the analytic phase curve calculations."""

    def test_phase_curve_static_single_wav(self, map):
        # Rotate the map and the source over time
        source = np.array([[np.sin(3 * t), np.cos(4 * t), np.sin(t)] 
                        for t in np.linspace(0, 5, 100)])
        theta = np.linspace(0, 180, 100)

        # Compute the flux analytically
        if map._reflected:
            flux = map.flux(theta=theta, source=source)
        else:
            flux = map.flux(theta=theta)
        flux /= np.nanmedian(flux)

        # Compute it numerically
        flux_num = np.nansum(map.render(res=50, theta=theta, 
                             source=source), axis=(1, 2))
        flux_num /= np.nanmedian(flux_num)
        
        # Compare
        assert np.allclose(flux_num, flux, atol=1e-2, rtol=1e-2)