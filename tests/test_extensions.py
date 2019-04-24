"""Test the custom extensions."""
import numpy as np
from starry import MAP, RAxisAngle


def test_MAP():
    nt = 10
    nr = 5
    A = np.random.randn(nt, nr)
    flux = np.random.randn(nt)
    C = 1.0
    L = 1.0
    yhat, yvar = MAP(A, L, C, flux, return_cov=True)

    yhat_, yvar_ = MAP(A, np.ones(nr) * L, C, flux, return_cov=True)
    assert np.allclose(yhat, yhat_)
    assert np.allclose(yvar, yvar_)

    yhat_, yvar_ = MAP(A, np.diag(np.ones(nr) * L), C, flux, return_cov=True)
    assert np.allclose(yhat, yhat_)
    assert np.allclose(yvar, yvar_)

    yhat_, yvar_ = MAP(A, L, np.ones(nt) * C, flux, return_cov=True)
    assert np.allclose(yhat, yhat_)
    assert np.allclose(yvar, yvar_)

    yhat_, yvar_ = MAP(A, L, np.diag(np.ones(nt)) * C, flux, return_cov=True)
    assert np.allclose(yhat, yhat_)
    assert np.allclose(yvar, yvar_)


def test_RAxisAngle():
    x = np.array([1, 0, 0], dtype=float)
    y = np.array([0, 1, 0], dtype=float)
    z = np.array([0, 0, 1], dtype=float)
    assert np.allclose(np.dot(RAxisAngle(z, 90), x), y)
    assert np.allclose(np.dot(RAxisAngle(z, 90), y), -x)
    assert np.allclose(np.dot(RAxisAngle(x, 90), y), z)
    assert np.allclose(np.dot(RAxisAngle(x, 90), z), -y)
    assert np.allclose(np.dot(RAxisAngle(y, 90), x), -z)
    assert np.allclose(np.dot(RAxisAngle(y, 90), z), x)