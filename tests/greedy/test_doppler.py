import starry
import numpy as np
from scipy.linalg import block_diag
import pytest


@pytest.fixture
def map():
    map = starry.DopplerMap(ydeg=10, nt=3, nc=2, nw=199, veq=50000)
    map.load(["spot", "earth"], force_psd=True)
    yield map


@pytest.fixture
def random():
    yield np.random.default_rng(0)


def test_flux(map):
    """
    Test that our various implementations of the flux
    yield identical results.

    """
    flux1 = map.flux(mode="convdot")
    flux2 = map.flux(mode="conv")
    flux3 = map.flux(mode="design")
    assert np.allclose(flux1, flux2)
    assert np.allclose(flux1, flux3)


def test_dot(map, random):
    """
    Test that our fast dot product method yields the same result as
    instantiating the full design matrix and dotting it in.

    """
    D = map.design_matrix().todense()
    matrix = random.normal(size=(len(map.wav0) * map.Ny, 5))
    product1 = D @ matrix
    product2 = map.dot(matrix)
    assert np.allclose(product1, product2)


def test_D_fixed_spectrum(map, random):
    """
    Test that our fast method for computing the design matrix
    for fixed input spectrum yields the same result as instantiating
    the full design matrix and dotting the spectral block matrix in.

    """
    DS = np.zeros((map.nt * len(map.wav), 0))
    D = map.design_matrix().todense()
    for k in range(map.nc):
        S = block_diag(
            *[map.spectrum[k].reshape(-1, 1) for n in range(map.Ny)]
        )
        DS = np.hstack((DS, D @ S))
    DS_fast = map.design_matrix(fix_spectrum=True)
    assert np.allclose(DS, DS_fast)
