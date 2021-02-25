import numpy as np
import starry
import matplotlib.pyplot as plt
from datetime import datetime
import pytest
from scipy.interpolate import interp1d
from tqdm import tqdm


@pytest.mark.parametrize(
    "xs,ys,zs,source_npts",
    [
        [0, 1, 1, 1],
        [-1, 0, 1, 1],
        [0.5, 1, -0.5, 1],
        [-0.5, -0.5, -0.5, 1],
        [0.5, -0.5, 0.5, 1],
        [1e-08, 0, 1, 1],  # almost noon
        [0, 0, 1, 1],  # exactly noon
        [0, 1, 1, 300],
    ],
)
def test_X(
    xs,
    ys,
    zs,
    source_npts,
    theta=0,
    ro=0.1,
    res=300,
    ydeg=2,
    tol=1e-3,
    plot=False,
):
    # Params
    npts = 250
    xo = np.linspace(-1.5, 1.5, npts)
    yo = np.linspace(-0.3, 0.5, npts)
    theta = 0
    ro = 0.1
    res = 300
    ydeg = 2
    tol = 1e-3

    # Instantiate
    map = starry.Map(ydeg=ydeg, reflected=True, source_npts=source_npts)

    # Analytic
    X = map.amp * map.design_matrix(
        xs=xs, ys=ys, zs=zs, theta=theta, xo=xo, yo=yo, ro=ro
    )

    # Numerical
    (lat, lon), (x, y, z) = map.ops.compute_ortho_grid(res)
    image = np.zeros((map.Ny, res * res))
    image[0] = map.render(theta=theta, xs=xs, ys=ys, zs=zs, res=res).flatten()
    n = 1
    for l in range(1, map.ydeg + 1):
        for m in range(-l, l + 1):
            map.reset()
            map[l, m] = 1
            image[n] = (
                map.render(theta=theta, xs=xs, ys=ys, zs=zs, res=res).flatten()
            ) - image[0]
            n += 1
    X_num = np.zeros_like(X)
    for k in range(len(xo)):
        idx = (x - xo[k]) ** 2 + (y - yo[k]) ** 2 > ro ** 2
        for n in range(map.Ny):
            X_num[k, n] = np.nansum(image[n][idx])
    X_num *= 4 / res ** 2

    # Plot
    if plot:

        fig, ax = plt.subplots(
            ydeg + 1, 2 * ydeg + 1, figsize=(9, 6), sharex=True, sharey=True
        )
        for axis in ax.flatten():
            axis.set_xticks([])
            axis.set_yticks([])
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.spines["bottom"].set_visible(False)
            axis.spines["left"].set_visible(False)
        n = 0
        for i, l in enumerate(range(ydeg + 1)):
            for j, m in enumerate(range(-l, l + 1)):
                j += ydeg - l
                med = np.median(X_num[:, n])
                ax[i, j].plot(X[:, n] - med, lw=2)
                ax[i, j].plot(X_num[:, n] - med, lw=1)
                n += 1

        fig.savefig(
            "test_X_{}.pdf".format(datetime.now().strftime("%d%m%Y%H%M%S")),
            bbox_inches="tight",
        )
        plt.close()

    # Compare
    diff = (X - X_num).flatten()
    assert np.max(np.abs(diff)) < tol


def test_inference():
    """
    Test inference on a problem with phase curve + occultations in reflected light.
    The orbital parameters here are contrived to ensure there's no null space;
    note the tiny observational uncertainty as well. Given this setup, a posterior
    map draw should look *very* similar to the true map.

    """
    # Orbital/geometric parameters
    npts = 50000
    t = np.linspace(0, 1, npts)
    porb = 0.19
    prot = 0.12
    rorb = 50
    ro = 38.0
    yo = np.sin(2 * np.pi / porb * t + 0.5)
    xo = np.cos(2 * np.pi / porb * t)
    zo = np.sin(2 * np.pi / porb * t)
    amp = rorb / np.sqrt(xo ** 2 + yo ** 2 + zo ** 2)
    xo *= amp
    yo *= amp
    zo *= amp
    theta = 360.0 / prot * t
    xs = np.sin(7 * np.pi * t)
    ys = np.cos(5 * np.pi * t)
    zs = 5
    kwargs = dict(xs=xs, ys=ys, zs=zs, theta=theta, xo=xo, yo=yo, zo=zo, ro=ro)

    # Generate a synthetic dataset
    map = starry.Map(ydeg=10, reflected=True)
    map.load("earth")
    img0 = map.render(projection="rect", illuminate=False)
    flux0 = map.flux(**kwargs)
    err = 1e-9
    np.random.seed(3)
    flux = flux0 + np.random.randn(npts) * err

    # Solve the linear problem & draw a sample
    map.set_data(flux, C=err ** 2)
    map.set_prior(L=1e-4)
    map.solve(**kwargs)
    map.draw()
    img = map.render(projection="rect", illuminate=False)

    # Verify we recovered the map
    assert np.allclose(img, img0, atol=1e-4)


@pytest.mark.parametrize(
    "b,theta,ro",
    [
        [0.25, np.pi / 3, 0.3],
        [-0.25, np.pi / 3, 0.3],
        [0.25, -np.pi / 3, 0.3],
        [-0.25, -np.pi / 3, 0.3],
        [0.25, 2 * np.pi / 3, 0.3],
        [-0.25, 2 * np.pi / 3, 0.3],
        [0.25, 4 * np.pi / 3, 0.3],
        [-0.25, 4 * np.pi / 3, 0.3],
        [0.5, np.pi / 2, 1.0],
        [0.0, 0.0, 0.5],
        [0.5, 0.0, 0.1],
        [1.0 - 1e-3, 0.0, 0.5],
        [-1.0 + 1e-3, 0.0, 0.5],
        [-1.0, 0.0, 0.5],
        [1.0, 0.0, 0.5],
        [0.25, np.pi / 2, 0.5],
    ],
)
def test_lightcurve(b, theta, ro, ydeg=1, ns=1000, nb=50, res=999, plot=False):

    # Array over full occultation, including all singularities
    xo = 0.0
    yo = np.linspace(0, 1 + ro, ns, endpoint=True)
    for pt in [ro, 1, 1 - ro, b + ro]:
        if pt >= 0:
            yo[np.argmin(np.abs(yo - pt))] = pt
    if theta == 0:
        xs = 0
        ys = 1
    else:
        xs = 0.5
        ys = -xs / np.tan(theta)
    rxy2 = xs ** 2 + ys ** 2
    if b == 0:
        zs = 0
    elif b == 1:
        zs = -1
        xs = 0
        ys = 0
    elif b == -1:
        zs = 1
        xs = 0
        ys = 0
    else:
        zs = -np.sign(b) * np.sqrt(rxy2 / (b ** -2 - 1))

    # Compute analytic
    map = starry.Map(ydeg=ydeg, reflected=True)
    map[1:, :] = 1
    flux = map.flux(xs=xs, ys=ys, zs=zs, xo=xo, yo=yo, ro=ro)

    # Compute numerical
    flux_num = np.zeros_like(yo) * np.nan
    computed = np.zeros(ns, dtype=bool)
    (lat, lon), (x, y, z) = map.ops.compute_ortho_grid(res)
    img = map.render(xs=xs, ys=ys, zs=zs, res=res).flatten()
    for i, yoi in tqdm(enumerate(yo), total=len(yo)):
        if (i == 0) or (i == ns - 1) or (i % (ns // nb) == 0):
            idx = (x - xo) ** 2 + (y - yoi) ** 2 > ro ** 2
            flux_num[i] = np.nansum(img[idx]) * 4 / res ** 2
            computed[i] = True

    # Interpolate over numerical result
    f = interp1d(yo[computed], flux_num[computed], kind="cubic")
    flux_num_interp = f(yo)

    # Plot
    if plot:
        fig = plt.figure()
        plt.plot(yo, flux, "C0-", label="starry", lw=2)
        plt.plot(yo, flux_num, "C1o", label="brute")
        plt.plot(yo, flux_num_interp, "C1-", lw=1)
        plt.legend(loc="best")
        plt.xlabel("impact parameter")
        plt.ylabel("flux")
        fig.savefig(
            "test_lightcurve[{}-{}-{}].pdf".format(b, theta, ro),
            bbox_inches="tight",
        )
        plt.close()

    # Compare with very lax tolerance; we're mostly looking
    # for gross outliers
    diff = np.abs(flux - flux_num_interp)
    assert np.max(diff) < 0.001


@pytest.mark.parametrize(
    "b,theta,bo,ro",
    [
        #
        # Occultor does not touch the terminator
        #
        [0.5, 0.1, 1.2, 0.1],
        [0.5, 0.1, 0.1, 1.2],
        [0.5, 0.1, 0.8, 0.1],
        [0.5, 0.1, 0.9, 0.2],
        [0.5, np.pi + 0.1, 0.8, 0.1],
        [0.5, np.pi + 0.1, 0.9, 0.2],
        [0.5, 0.1, 0.5, 1.25],
        [0.5, np.pi + 0.1, 0.5, 1.25],
        #
        # Occultations involving all three primitive integrals
        #
        [0.4, np.pi / 3, 0.5, 0.7],
        [0.4, 2 * np.pi - np.pi / 3, 0.5, 0.7],
        [0.4, np.pi / 2, 0.5, 0.7],
        [0.4, np.pi / 2, 1.0, 0.2],
        [0.00001, np.pi / 2, 0.5, 0.7],
        [0, np.pi / 2, 0.5, 0.7],
        [0.4, -np.pi / 2, 0.5, 0.7],
        [-0.4, np.pi / 3, 0.5, 0.7],
        [-0.4, 2 * np.pi - np.pi / 3, 0.5, 0.7],
        [-0.4, np.pi / 2, 0.5, 0.7],
        #
        # Occultations involving only P and T
        #
        [0.4, np.pi / 6, 0.3, 0.3],
        [0.4, np.pi + np.pi / 6, 0.1, 0.6],
        [0.4, np.pi + np.pi / 3, 0.1, 0.6],
        [0.4, np.pi / 6, 0.6, 0.5],
        [0.4, -np.pi / 6, 0.6, 0.5],
        [0.4, 0.1, 2.2, 2.0],
        [0.4, -0.1, 2.2, 2.0],
        [0.4, np.pi + np.pi / 6, 0.3, 0.8],
        [0.75, np.pi + 0.1, 4.5, 5.0],
        [-0.95, 0.0, 2.0, 2.5],
        [-0.1, np.pi / 6, 0.6, 0.75],
        [-0.5, np.pi, 0.8, 0.5],
        [-0.1, 0.0, 0.5, 1.0],
        #
        # Occultations involving three points of intersection with the terminator
        #
        [
            0.5488316824842527,
            4.03591586925189,
            0.34988513192814663,
            0.7753986686719786,
        ],
        [
            0.5488316824842527,
            2 * np.pi - 4.03591586925189,
            0.34988513192814663,
            0.7753986686719786,
        ],
        [
            -0.5488316824842527,
            4.03591586925189 - np.pi,
            0.34988513192814663,
            0.7753986686719786,
        ],
        [
            -0.5488316824842527,
            2 * np.pi - (4.03591586925189 - np.pi),
            0.34988513192814663,
            0.7753986686719786,
        ],
        #
        # Occultations involving four points of intersection with the terminator
        #
        [0.5, np.pi, 0.99, 1.5],
        [-0.5, 0.0, 0.99, 1.5],
        #
        # Miscellaneous edge cases
        #
        [0.5, np.pi, 1.0, 1.5],
        [0.5, 2 * np.pi - np.pi / 4, 0.4, 0.4],
        [0.5, 2 * np.pi - np.pi / 4, 0.3, 0.3],
        [-0.25, 4 * np.pi / 3, 0.3, 0.3],
    ],
)
def test_cases(b, theta, bo, ro, ydeg=1, res=999):

    # Array over full occultation, including all singularities
    xo = 0.0
    yo = bo
    if theta == 0:
        xs = 0
        ys = 1
    else:
        xs = 0.5
        ys = -xs / np.tan(theta)
    rxy2 = xs ** 2 + ys ** 2
    if b == 0:
        zs = 0
    elif b == 1:
        zs = -1
        xs = 0
        ys = 0
    elif b == -1:
        zs = 1
        xs = 0
        ys = 0
    else:
        zs = -np.sign(b) * np.sqrt(rxy2 / (b ** -2 - 1))

    # Compute analytic
    map = starry.Map(ydeg=ydeg, reflected=True)
    map[1:, :] = 1
    flux = map.flux(xs=xs, ys=ys, zs=zs, xo=xo, yo=yo, ro=ro)

    # Compute numerical
    (lat, lon), (x, y, z) = map.ops.compute_ortho_grid(res)
    img = map.render(xs=xs, ys=ys, zs=zs, res=res).flatten()
    idx = (x - xo) ** 2 + (y - yo) ** 2 > ro ** 2
    flux_num = np.nansum(img[idx]) * 4 / res ** 2

    # Compare with very lax tolerance; we're mostly looking
    # for gross outliers
    diff = np.abs(flux - flux_num)
    assert diff < 0.001


def test_theta_poles(res=500, tol=1e-3):
    """Test cases near the poles for theta."""
    # Settings
    ydeg = 10
    zs = -0.25
    xo = 0.0
    yo = 0.35
    ro = 0.25
    n = 5

    # Compare
    map = starry.Map(ydeg, reflected=True)
    map[ydeg, :] = 1
    x = np.array([0.0, 0.5, 1.0, 1.5, 2.0]).reshape(-1, 1) * np.pi
    dx = np.concatenate(
        (-np.logspace(-15, -5, n)[::-1], [0], np.logspace(-15, -5, n))
    ).reshape(1, -1)
    theta = (x + dx).reshape(-1)
    (lat, lon), (x, y, z) = map.ops.compute_ortho_grid(res)
    err = np.zeros_like(theta)
    for i in range(len(theta)):
        if theta[i] == 0:
            xs = 0
            ys = 1
        else:
            xs = 0.5
            ys = -xs / np.tan(theta[i])
        flux = map.flux(xs=xs, ys=ys, zs=zs, xo=xo, yo=yo, ro=ro)
        img = map.render(xs=xs, ys=ys, zs=zs, res=res).flatten()
        idx = (x - xo) ** 2 + (y - yo) ** 2 > ro ** 2
        flux_num = np.nansum(img[idx]) * 4 / res ** 2
        err[i] = np.max(np.abs(flux - flux_num))
    assert np.all(err < tol)


# BROKEN: Figure out why the root finder fails here.
@pytest.mark.xfail
def test_root_finder():
    """
    Test cases that cause the root finder to fail.

    """
    map = starry.Map(reflected=True)
    map.ops._sT.func([-0.358413], [-1.57303], [55.7963], 54.8581, 0.0)


# BROKEN: Figure this out
@pytest.mark.xfail
def test_bad_case():
    """
    Test pathological wrong case identification.

    """
    map = starry.Map(reflected=True)

    # These values lead to a (very) wrong flux
    theta0 = -0.0409517311212404
    b0 = -0.83208413089546
    bo0 = 12.073565287605442
    ro = 12.155639360414618

    # Perturb theta in the vicinity of theta0
    delta = np.linspace(0, 1e-6, 100)
    theta = np.concatenate((theta0 - delta[::-1], theta0 + delta))

    # Compute the flux
    b = b0 * np.ones_like(theta)
    bo = bo0 * np.ones_like(theta)
    sT, *_ = map.ops._sT.func(b, theta, bo, ro, 0.0)
    flux = sT[:, 0]

    # DEBUG
    # plt.plot(theta, flux)
    # plt.show()

    # Check that it's approximately constant over the range
    assert np.allclose(flux, flux[0])
