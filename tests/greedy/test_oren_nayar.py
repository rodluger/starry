import numpy as np
import matplotlib.pyplot as plt
import pytest
import starry


def test_terminator_continuity(plot=False):
    """
    Ensure the Oren-Nayar intensity is continuous across
    the day/night boundary.

    """
    # Simple map
    map = starry.Map(reflected=True)

    # Find the terminator latitude
    ys = 1
    zs = 1
    b = -zs / np.sqrt(ys ** 2 + zs ** 2)
    lat0 = np.arcsin(b) * 180 / np.pi

    if plot:

        # Latitude array spanning the terminator
        delta = 1
        lat = np.linspace(lat0 - delta, lat0 + delta, 1000)

        # Lambertian intensity
        map.roughness = 0
        I_lamb = map.intensity(lat=lat, lon=0, xs=0, ys=ys, zs=zs).reshape(-1)

        # Oren-Nayar intensity
        map.roughness = 90
        I_on94 = map.intensity(lat=lat, lon=0, xs=0, ys=ys, zs=zs).reshape(-1)

        # View it
        plt.plot(lat, I_lamb)
        plt.plot(lat, I_on94)
        plt.xlabel("lat")
        plt.ylabel("I")
        plt.show()

    # Ensure there's a negligible jump across the terminator
    eps = 1e-8
    lat = np.array([lat0 - eps, lat0 + eps])
    map.roughness = 90
    diff = np.diff(
        map.intensity(lat=lat, lon=0, xs=0, ys=ys, zs=zs).reshape(-1)
    )[0]
    assert np.abs(diff) < 1e-10, np.abs(diff)


def test_half_phase_discontinuity(plot=False):
    """
    Ensure the Oren-Nayar intensity at a point is continuous
    as we move across half phase (b = 0).

    """
    # Simple map
    map = starry.Map(reflected=True)

    if plot:

        # From crescent to gibbous
        eps = 0.1
        zs = np.linspace(-eps, eps, 1000)

        # Oren-Nayar intensity
        map.roughness = 90
        I_on94 = map.intensity(lat=60, lon=0, xs=0, ys=1, zs=zs).reshape(-1)

        # View it
        plt.plot(-zs, I_on94)
        plt.xlabel("b")
        plt.ylabel("I")
        plt.show()

    # Ensure there's a negligible jump across the terminator
    eps = 1e-8
    zs = np.array([-eps, eps])
    map.roughness = 90
    diff = np.diff(
        map.intensity(lat=60, lon=0, xs=0, ys=1, zs=zs).reshape(-1)
    )[0]
    assert np.abs(diff) < 1e-8, np.abs(diff)


def test_approximation(plot=False):
    """
    Test our polynomial approximation to the Oren-Nayar intensity.

    """
    # Simple map
    map = starry.Map(reflected=True)

    # Approximate and exact intensities
    map.roughness = 90
    img_approx = map.render(xs=1, ys=2, zs=3)
    img_exact = map.render(xs=1, ys=2, zs=3, on94_exact=True)
    img_diff = img_exact - img_approx
    diff = img_diff.reshape(-1)

    mu = np.nanmean(diff)
    std = np.nanstd(diff)
    maxabs = np.nanmax(np.abs(diff))

    if plot:
        fig, ax = plt.subplots(1, 3, figsize=(14, 2.5))
        im = ax[0].imshow(
            img_exact,
            origin="lower",
            extent=(-1, 1, -1, 1),
            vmin=0,
            vmax=np.nanmax(img_exact),
        )
        plt.colorbar(im, ax=ax[0])
        im = ax[1].imshow(
            img_approx,
            origin="lower",
            extent=(-1, 1, -1, 1),
            vmin=0,
            vmax=np.nanmax(img_exact),
        )
        plt.colorbar(im, ax=ax[1])
        im = ax[2].imshow(img_diff, origin="lower", extent=(-1, 1, -1, 1))
        plt.colorbar(im, ax=ax[2])

        fig = plt.figure()
        plt.hist(diff, bins=50)
        plt.xlabel("diff")
        plt.show()

    assert np.abs(mu) < 1e-5, np.abs(mu)
    assert std < 1e-3, std
    assert maxabs < 1e-2, maxabs


if __name__ == "__main__":
    starry.config.lazy = False
    test_terminator_continuity()
    test_half_phase_discontinuity()
    test_approximation()
