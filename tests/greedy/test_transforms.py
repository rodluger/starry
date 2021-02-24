import numpy as np
import starry


def test_latlon_grid():
    # Just check that these don't cause errors
    map = starry.Map(10)
    lat, lon = map.get_latlon_grid(projection="rect")
    lat, lon = map.get_latlon_grid(projection="ortho")
    lat, lon = map.get_latlon_grid(projection="moll")


def test_pixel_transforms():
    map = starry.Map(10)
    lat, lon, Y2P, P2Y, Dx, Dy = map.get_pixel_transforms()

    # Check that the back-and-forth transform is the identity (ish)
    assert np.max(np.abs(P2Y @ Y2P - np.eye(map.Ny))) < 1e-6

    # Just check that the derivatives are finite
    assert not np.isnan(np.sum(Dx) + np.sum(Dy))
