import starry
import numpy as np
from scipy.linalg import block_diag
from scipy.sparse import diags
import pytest


@pytest.fixture(scope="module", params=[1, 2])
def map(request):
    map = starry.DopplerMap(ydeg=10, nt=3, nc=request.param, veq=50000)
    map.load(["spot", "earth"], force_psd=True)
    yield map


@pytest.fixture(scope="function")
def random():
    yield np.random.default_rng(0)


def test_flux(map):
    """
    Test that our various implementations of the flux
    yield identical results.

    """
    flux1 = map.flux(mode="convdot")
    flux2 = map.flux(mode="dotconv")
    flux3 = map.flux(mode="conv")
    flux4 = map.flux(mode="design")
    assert np.allclose(flux1, flux2)
    assert np.allclose(flux1, flux3)
    assert np.allclose(flux1, flux4)


def test_dot_matrix(map, random):
    """
    Test that our fast dot product method yields the same result as
    instantiating the full design matrix and dotting it in.

    """
    D = map.design_matrix().todense()
    matrix = random.normal(size=(map.nw0_ * map.Ny, 5))
    product1 = D @ matrix
    product2 = map.dot(matrix)
    assert np.allclose(product1, product2)


def test_dot_vector(map, random):
    """
    Test that our fast dot product method yields the same result as
    instantiating the full design matrix and dotting it in.

    """
    D = np.array(map.design_matrix().todense())
    vector = random.normal(size=(map.nw0_ * map.Ny,))
    product1 = (D @ vector).reshape(-1)
    product2 = map.dot(vector).reshape(-1)
    assert np.allclose(product1, product2)


def test_D_fixed_spectrum(map, random):
    """
    Test that our fast method for computing the design matrix
    for fixed input spectrum yields the same result as instantiating
    the full design matrix and dotting the spectral block matrix `S` in.

    """
    # Compute it manually
    DS = np.zeros((map.nt * map.nw, 0))
    D = map.design_matrix().todense()
    for k in range(map.nc):
        S = block_diag(
            *[map.spectrum_[k].reshape(-1, 1) for n in range(map.Ny)]
        )
        DS = np.hstack((DS, D @ S))

    # Compute it with starry
    DS_fast = map.design_matrix(fix_spectrum=True)

    # Check that the matrices are the same
    assert np.allclose(DS, DS_fast)

    # Check that this procedure yields the correct flux
    flux1 = (DS_fast @ map.y.T.reshape(-1)).reshape(map.nt, map.nw)
    flux2 = (D @ map.spectral_map).reshape(map.nt, map.nw)
    assert np.allclose(flux1, flux2)


def test_D_fixed_map(map, random):
    """
    Test that our fast method for computing the design matrix
    for fixed input map yields the same result as instantiating
    the full design matrix and dotting the map block matrix `Y` in.

    """
    # Compute it manually
    DY = np.zeros((map.nt * map.nw, 0))
    D = map.design_matrix().todense()
    if map.nc == 1:
        y = np.reshape(map.y, [-1, 1])
    else:
        y = map.y
    for k in range(map.nc):
        Y = diags(
            [np.ones(map.nw0_) * y[n, k] for n in range(map.Ny)],
            offsets=-np.arange(0, map.Ny) * map.nw0_,
            shape=(map.Ny * map.nw0_, map.nw0_),
        ).todense()
        DY = np.hstack((DY, D @ Y))

    # Compute it with starry
    DY_fast = map.design_matrix(fix_map=True)

    # Check that the matrices are the same
    assert np.allclose(DY, DY_fast)

    # Check that this procedure yields the correct flux
    flux1 = (DY_fast @ map.spectrum_.reshape(-1)).reshape(map.nt, map.nw)
    flux2 = (D @ map.spectral_map).reshape(map.nt, map.nw)
    assert np.allclose(flux1, flux2)
