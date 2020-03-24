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
_s = theano.function([_b, _theta, _bo, _ro], map.ops.sT(_b, _theta, _bo, _ro))


def s(b, theta, bo, ro, n=0):
    if hasattr(ro, "__len__"):
        assert not (
            hasattr(b, "__len__")
            or hasattr(theta, "__len__")
            or hasattr(bo, "__len__")
        )
        return [_s([b], [theta], [bo], ro[i])[0, n] for i in range(len(ro))]
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
        return _s(b, theta, bo, ro)[:, n]


_dsdb = [
    theano.function(
        [_b, _theta, _bo, _ro],
        tt.grad(map.ops.sT(_b, _theta, _bo, _ro)[0, n], _b),
    )
    for n in range(map.Ny)
]
_dsdtheta = [
    theano.function(
        [_b, _theta, _bo, _ro],
        tt.grad(map.ops.sT(_b, _theta, _bo, _ro)[0, n], _theta),
    )
    for n in range(map.Ny)
]
_dsdbo = [
    theano.function(
        [_b, _theta, _bo, _ro],
        tt.grad(map.ops.sT(_b, _theta, _bo, _ro)[0, n], _bo),
    )
    for n in range(map.Ny)
]
_dsdro = [
    theano.function(
        [_b, _theta, _bo, _ro],
        tt.grad(map.ops.sT(_b, _theta, _bo, _ro)[0, n], _ro),
    )
    for n in range(map.Ny)
]


def dsdb(b, theta, bo, ro, n=0):
    b = np.atleast_1d(b)
    assert not hasattr(theta, "__len__")
    assert not hasattr(bo, "__len__")
    assert not hasattr(ro, "__len__")
    return np.array(
        [_dsdb[n]([b[i]], [theta], [bo], ro) for i in range(len(b))]
    )


def dsdtheta(b, theta, bo, ro, n=0):
    theta = np.atleast_1d(theta)
    assert not hasattr(b, "__len__")
    assert not hasattr(bo, "__len__")
    assert not hasattr(ro, "__len__")
    return np.array(
        [_dsdtheta[n]([b], [theta[i]], [bo], ro) for i in range(len(theta))]
    )


def dsdbo(b, theta, bo, ro, n=0):
    bo = np.atleast_1d(bo)
    assert not hasattr(b, "__len__")
    assert not hasattr(theta, "__len__")
    assert not hasattr(ro, "__len__")
    return np.array(
        [_dsdbo[n]([b], [theta], [bo[i]], ro) for i in range(len(bo))]
    )


def dsdro(b, theta, bo, ro, n=0):
    ro = np.atleast_1d(ro)
    assert not hasattr(b, "__len__")
    assert not hasattr(theta, "__len__")
    assert not hasattr(bo, "__len__")
    return np.array(
        [_dsdro[n]([b], [theta], [bo], ro[i]) for i in range(len(ro))]
    )


def grad(b, theta, bo, ro, n=0):
    if hasattr(b, "__len__"):
        wrt = b
    elif hasattr(theta, "__len__"):
        wrt = theta
    elif hasattr(bo, "__len__"):
        wrt = bo
    elif hasattr(ro, "__len__"):
        wrt = ro
    else:
        assert False
    return np.gradient(s(b, theta, bo, ro, n=n), edge_order=2) / np.gradient(
        wrt, edge_order=2
    )


def test_derivs(n=1, npts=10000, atol=1e-5, plot=False):

    if plot:
        fig, ax = plt.subplots(3, 4, figsize=(16, 7))

    # b gradient
    theta = 0.51
    bo = 0.75
    ro = 0.1
    b = np.linspace(-1, 1, npts)
    g1 = grad(b, theta, bo, ro, n=n)
    g2 = dsdb(b, theta, bo, ro, n=n).flatten()
    # Pad the edges (numerical gradient isn't great)
    assert np.allclose(g1[200:-50], g2[200:-50], atol=atol), "error in b"

    if plot:
        ax[0, 0].plot(b, s(b, theta, bo, ro, n=n))
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
    theta = np.linspace(-np.pi, np.pi, npts)
    g1 = grad(b, theta, bo, ro, n=n)
    g2 = dsdtheta(b, theta, bo, ro, n=n).flatten()
    assert np.allclose(g1, g2, atol=atol), "error in theta"

    if plot:
        ax[0, 1].plot(theta, s(b, theta, bo, ro, n=n))
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
    bo = np.linspace(0, 1.5, npts)
    g1 = grad(b, theta, bo, ro, n=n)
    g2 = dsdbo(b, theta, bo, ro, n=n).flatten()
    assert np.allclose(g1, g2, atol=atol), "error in bo"

    if plot:
        ax[0, 2].plot(bo, s(b, theta, bo, ro, n=n))
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
    ro = np.linspace(0.001, 1.5, npts)
    g1 = grad(b, theta, bo, ro, n=n)
    g2 = dsdro(b, theta, bo, ro, n=n).flatten()
    assert np.allclose(g1, g2, atol=atol), "error in ro"

    if plot:
        ax[0, 3].plot(ro, s(b, theta, bo, ro, n=n))
        ax[1, 3].plot(ro, g1, lw=2)
        ax[1, 3].plot(ro, g2, lw=1)
        ax[1, 3].set_ylim(-1.5, 1.0)
        ax[2, 3].plot(ro, np.log10(np.abs(g1 - g2)), "k")
        ax[2, 3].set_xlabel("ro")
        for axis in ax[:, 3]:
            axis.set_xlim(0.001, 1.5)

        for axis in ax[2, :]:
            axis.set_ylim(-6, 0)
        plt.show()


if __name__ == "__main__":
    test_derivs(plot=True)
