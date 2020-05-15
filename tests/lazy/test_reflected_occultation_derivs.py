import theano
import theano.tensor as tt
import numpy as np
import starry
import matplotlib.pyplot as plt
import pytest


map = starry.Map(ydeg=1, reflected=True)
_b = tt.dvector("b")
_theta = tt.dvector("theta")
_bo = tt.dvector("bo")
_ro = tt.dscalar("ro")
_sigr = tt.dscalar("sigr")
_s = theano.function(
    [_b, _theta, _bo, _ro, _sigr], map.ops.sT(_b, _theta, _bo, _ro, _sigr)
)


def s(b, theta, bo, ro, sigr, n=0):
    if hasattr(ro, "__len__"):
        assert not (
            hasattr(b, "__len__")
            or hasattr(theta, "__len__")
            or hasattr(bo, "__len__")
            or hasattr(sigr, "__len__")
        )
        return [
            _s([b], [theta], [bo], ro[i], sigr)[0, n] for i in range(len(ro))
        ]
    elif hasattr(sigr, "__len__"):
        assert not (
            hasattr(b, "__len__")
            or hasattr(theta, "__len__")
            or hasattr(bo, "__len__")
            or hasattr(ro, "__len__")
        )
        return [
            _s([b], [theta], [bo], ro, sigr[i])[0, n] for i in range(len(sigr))
        ]
    else:
        assert (
            hasattr(b, "__len__")
            or hasattr(theta, "__len__")
            or hasattr(bo, "__len__")
        )
        shaper = np.zeros_like(b) + np.zeros_like(bo) + np.zeros_like(theta)
        b += shaper
        theta += shaper
        bo += shaper
        return _s(b, theta, bo, ro, sigr)[:, n]


_dsdb = [
    theano.function(
        [_b, _theta, _bo, _ro, _sigr],
        tt.grad(map.ops.sT(_b, _theta, _bo, _ro, _sigr)[0, n], _b),
    )
    for n in range(map.Ny)
]
_dsdtheta = [
    theano.function(
        [_b, _theta, _bo, _ro, _sigr],
        tt.grad(map.ops.sT(_b, _theta, _bo, _ro, _sigr)[0, n], _theta),
    )
    for n in range(map.Ny)
]
_dsdbo = [
    theano.function(
        [_b, _theta, _bo, _ro, _sigr],
        tt.grad(map.ops.sT(_b, _theta, _bo, _ro, _sigr)[0, n], _bo),
    )
    for n in range(map.Ny)
]
_dsdro = [
    theano.function(
        [_b, _theta, _bo, _ro, _sigr],
        tt.grad(map.ops.sT(_b, _theta, _bo, _ro, _sigr)[0, n], _ro),
    )
    for n in range(map.Ny)
]
_dsdsigr = [
    theano.function(
        [_b, _theta, _bo, _ro, _sigr],
        tt.grad(map.ops.sT(_b, _theta, _bo, _ro, _sigr)[0, n], _sigr),
    )
    for n in range(map.Ny)
]


def dsdb(b, theta, bo, ro, sigr, n=0):
    b = np.atleast_1d(b)
    assert not hasattr(theta, "__len__")
    assert not hasattr(bo, "__len__")
    assert not hasattr(ro, "__len__")
    assert not hasattr(sigr, "__len__")
    return np.array(
        [_dsdb[n]([b[i]], [theta], [bo], ro, sigr) for i in range(len(b))]
    )


def dsdtheta(b, theta, bo, ro, sigr, n=0):
    theta = np.atleast_1d(theta)
    assert not hasattr(b, "__len__")
    assert not hasattr(bo, "__len__")
    assert not hasattr(ro, "__len__")
    assert not hasattr(sigr, "__len__")
    return np.array(
        [
            _dsdtheta[n]([b], [theta[i]], [bo], ro, sigr)
            for i in range(len(theta))
        ]
    )


def dsdbo(b, theta, bo, ro, sigr, n=0):
    bo = np.atleast_1d(bo)
    assert not hasattr(b, "__len__")
    assert not hasattr(theta, "__len__")
    assert not hasattr(ro, "__len__")
    assert not hasattr(sigr, "__len__")
    return np.array(
        [_dsdbo[n]([b], [theta], [bo[i]], ro, sigr) for i in range(len(bo))]
    )


def dsdro(b, theta, bo, ro, sigr, n=0):
    ro = np.atleast_1d(ro)
    assert not hasattr(b, "__len__")
    assert not hasattr(theta, "__len__")
    assert not hasattr(bo, "__len__")
    assert not hasattr(sigr, "__len__")
    return np.array(
        [_dsdro[n]([b], [theta], [bo], ro[i], sigr) for i in range(len(ro))]
    )


def dsdsigr(b, theta, bo, ro, sigr, n=0):
    sigr = np.atleast_1d(sigr)
    assert not hasattr(b, "__len__")
    assert not hasattr(theta, "__len__")
    assert not hasattr(bo, "__len__")
    assert not hasattr(ro, "__len__")
    return np.array(
        [
            _dsdsigr[n]([b], [theta], [bo], ro, sigr[i])
            for i in range(len(sigr))
        ]
    )


def grad(b, theta, bo, ro, sigr, n=0):
    if hasattr(b, "__len__"):
        wrt = b
    elif hasattr(theta, "__len__"):
        wrt = theta
    elif hasattr(bo, "__len__"):
        wrt = bo
    elif hasattr(ro, "__len__"):
        wrt = ro
    elif hasattr(sigr, "__len__"):
        wrt = sigr
    else:
        assert False
    return np.gradient(
        s(b, theta, bo, ro, sigr, n=n), edge_order=2
    ) / np.gradient(wrt, edge_order=2)


def test_derivs(n=1, npts=10000, atol=1e-5, plot=False, throw=True):

    if plot:
        fig, ax = plt.subplots(3, 5, figsize=(16, 7))

    # b gradient
    theta = 0.51
    bo = 0.75
    ro = 0.1
    sigr = 0.0
    b = np.linspace(-1, 1, npts)
    g1 = grad(b, theta, bo, ro, sigr, n=n)
    g2 = dsdb(b, theta, bo, ro, sigr, n=n).flatten()
    # Pad the edges (numerical gradient isn't great)
    if throw:
        assert np.allclose(g1[200:-50], g2[200:-50], atol=atol), "error in b"

    if plot:
        ax[0, 0].plot(b, s(b, theta, bo, ro, sigr, n=n))
        ax[1, 0].plot(b, g1, lw=2)
        ax[1, 0].plot(b, g2, lw=1)
        ax[2, 0].plot(b, np.log10(np.abs(g1 - g2)), "k")
        ax[2, 0].set_xlabel("b")
        for axis in ax[:, 0]:
            axis.set_xlim(-1, 1)

    # theta gradient
    b = 0.51
    bo = 0.75
    ro = 0.1
    sigr = 30 * np.pi / 180
    theta = np.linspace(-np.pi, np.pi, npts)
    g1 = grad(b, theta, bo, ro, sigr, n=n)
    g2 = dsdtheta(b, theta, bo, ro, sigr, n=n).flatten()
    if throw:
        assert np.allclose(g1, g2, atol=atol), "error in theta"

    if plot:
        ax[0, 1].plot(theta, s(b, theta, bo, ro, sigr, n=n))
        ax[1, 1].plot(theta, g1, lw=2)
        ax[1, 1].plot(theta, g2, lw=1)
        ax[2, 1].plot(theta, np.log10(np.abs(g1 - g2)), "k")
        ax[2, 1].set_xlabel("theta")
        for axis in ax[:, 1]:
            axis.set_xlim(-np.pi, np.pi)

    # bo gradient
    b = 0.51
    theta = 0.49
    ro = 0.1
    sigr = 30 * np.pi / 180
    bo = np.linspace(0, 1.5, npts)
    g1 = grad(b, theta, bo, ro, sigr, n=n)
    g2 = dsdbo(b, theta, bo, ro, sigr, n=n).flatten()
    if throw:
        assert np.allclose(g1, g2, atol=atol), "error in bo"

    if plot:
        ax[0, 2].plot(bo, s(b, theta, bo, ro, sigr, n=n))
        ax[1, 2].plot(bo, g1, lw=2)
        ax[1, 2].plot(bo, g2, lw=1)
        ax[2, 2].plot(bo, np.log10(np.abs(g1 - g2)), "k")
        ax[2, 2].set_xlabel("bo")
        for axis in ax[:, 2]:
            axis.set_xlim(0, 1.5)

    # ro gradient
    b = 0.51
    theta = 0.49
    bo = 0.75
    sigr = 30 * np.pi / 180
    ro = np.linspace(0.001, 1.5, npts)
    g1 = grad(b, theta, bo, ro, sigr, n=n)
    g2 = dsdro(b, theta, bo, ro, sigr, n=n).flatten()
    if throw:
        assert np.allclose(g1, g2, atol=atol), "error in ro"

    if plot:
        ax[0, 3].plot(ro, s(b, theta, bo, ro, sigr, n=n))
        ax[1, 3].plot(ro, g1, lw=2)
        ax[1, 3].plot(ro, g2, lw=1)
        ax[1, 3].set_ylim(-1.5, 1.0)
        ax[2, 3].plot(ro, np.log10(np.abs(g1 - g2)), "k")
        ax[2, 3].set_xlabel("ro")
        for axis in ax[:, 3]:
            axis.set_xlim(0.001, 1.5)

        for axis in ax[2, :]:
            axis.set_ylim(-6, 0)

    # sigr gradient
    b = 0.51
    theta = 0.49
    bo = 0.75
    ro = 0.1
    sigr = np.linspace(0, 30 * np.pi / 180, npts)
    g1 = grad(b, theta, bo, ro, sigr, n=n)
    g2 = dsdsigr(b, theta, bo, ro, sigr, n=n).flatten()
    if throw:
        assert np.allclose(g1, g2, atol=atol), "error in sigr"

    if plot:
        ax[0, 4].plot(sigr, s(b, theta, bo, ro, sigr, n=n))
        ax[1, 4].plot(sigr, g1, lw=2)
        ax[1, 4].plot(sigr, g2, lw=1)
        ax[1, 4].set_ylim(-1.5, 1.0)
        ax[2, 4].plot(sigr, np.log10(np.abs(g1 - g2)), "k")
        ax[2, 4].set_xlabel("sigr")
        for axis in ax[:, 4]:
            axis.set_xlim(0, 30 * np.pi / 180)

        plt.show()


def test_abs_b_one():
    """Check derivs are finite when b=+/-1."""
    b = tt.dscalar()
    b.tag.test_value = -1.0
    map = starry.Map(reflected=True)

    def flux(b):
        return map.flux(zs=-b, ys=0)

    grad = theano.function([b], tt.grad(flux(b)[0], [b]))
    assert not np.isnan(grad(-1.0)[0]) and not np.isnan(grad(1.0)[0])


if __name__ == "__main__":
    test_abs_b_one()
