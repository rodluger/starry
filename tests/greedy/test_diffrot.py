# -*- coding: utf-8 -*-
"""
Test differential rotation.

"""
import starry
import numpy as np


def test_diffrot(visualize=False):
    """Test differential rotation on a map with a cosine-like meridional band."""
    map = starry.Map(ydeg=5, drorder=1)
    map.alpha = 0.1

    # This is a bit hacky: compute the Ylm expansion of cos(latitude)
    # then rotate it 90 degrees so it's cos(longitude) instead.
    # Expanding a longitudinal cosine directly doesn't work as well
    map.load(
        np.tile(
            np.cos(np.linspace(-np.pi, np.pi, 500, endpoint=False)), (1000, 1)
        ).T,
        nside=128,
    )
    map.rotate([0, 0, 1], 90)

    # Render the map at 5 phases
    theta = [0, 90, 180, 270, 360]
    res = 300
    images = map.render(projection="rect", theta=theta, res=res)
    lat = np.linspace(-90, 90, res, endpoint=False)
    lon = np.linspace(-180, 180, res, endpoint=False)

    # Compute the longitude of the maximum on the side
    # facing the observer; it's easiest if we just mask the far side
    images_masked = np.array(images)
    images_masked[:, :, : (res // 4)] = 0
    images_masked[:, :, -(res // 4) :] = 0
    lon_starry = [lon[np.argmax(img, axis=1)] for img in images_masked]

    # Compute the expected longitude of the maximum given
    # the linear differential rotation law
    lon_exact = [
        -theta_i * map.alpha * np.sin(lat * np.pi / 180) ** 2
        for theta_i in theta
    ]

    # Check that the expressions agree within the error of the expansion
    poles = (np.abs(lat) > 60) & (np.abs(lat) < 89)
    tropics = np.abs(lat) < 60
    for k in range(5):
        diff = np.abs(lon_exact[k] - lon_starry[k])

        # Poles should agree within 5 degrees
        assert np.all(diff[poles] < 5)

        # "Tropics" should agree within 2 degrees
        assert np.all(diff[tropics] < 2)

    # Finally, check that the intensity we get when calling
    # `map.intensity` agrees with that from `map.render`
    I1 = map.intensity(lat=lat[250], lon=lon, theta=360)
    I2 = images[-1][250, :]
    assert np.allclose(I1, I2)
