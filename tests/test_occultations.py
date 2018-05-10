"""Test occultation light curves."""
from starry import Map
import numpy as np
np.random.seed(1234)


def test_occultations():
    """Test occultation light curves."""
    # Let's do the l = 3 Earth
    m = Map(3)
    m.load_image('earth')

    # Rotate the map about a random axis
    ux = np.random.random()
    uy = np.random.random() * (1 - ux)
    uz = np.sqrt(1 - ux ** 2 - uy ** 2)
    axis = [ux, uy, uz]
    npts = 30
    theta = np.linspace(0, 2 * np.pi, npts, endpoint=False)

    # Small occultor
    ro = 0.3
    xo = np.linspace(-1 - ro - 0.1, 1 + ro + 0.1, npts)
    yo = 0

    # Analytical and numerical fluxes
    sF = np.array(m.flux(axis=axis, theta=theta, xo=xo, yo=yo, ro=ro))
    nF = np.array(m.flux_numerical(axis=axis, theta=theta, xo=xo, yo=yo, ro=ro,
                                   tol=1e-6))

    # Compute the (relative) error
    error = np.max(np.abs(sF - nF))

    # Our numerical integration scheme isn't the most accurate,
    # so let's be lenient here!
    assert error < 1e-2


if __name__ == "__main__":
    test_occultations()
