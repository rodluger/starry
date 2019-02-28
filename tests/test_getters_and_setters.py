"""Test coefficient getters/setters."""
import starry
import numpy as np
import pytest


@pytest.fixture(
    scope="class",
    params=[
        (None, None, False, False),
        (3, None, False, False),
        (None, 3, False, False),
        (None, None, True, False),
        (3, None, True, False),
        (None, 3, True, False),
        #(None, None, False, True),
        #(3, None, False, True),
        #(None, 3, False, True),
        #(None, None, True, True),
        #(3, None, True, True),
        #(None, 3, True, True),
    ],
)
def map(request):
    nw, nt, multi, reflected = request.param
    map = starry.Map(ydeg=5, udeg=2, nw=nw, nt=nt, 
                     multi=multi, reflected=reflected)
    return map


class TestGettersAndSetters:
    """Test the Map getters and setters."""

    def test_ylm_single_to_scalar(self, map):
        if map._spectral:
            map[1, 1, 0] = 7
            assert map[1, 1, 0] == 7
            assert map.y[3, 0] == 7
        elif map._temporal:
            map[1, 1, 0] = 7
            assert map[1, 1, 0] == 7
            assert map.y.reshape(map.nt, -1).transpose()[3, 0] == 7
        else:
            map[1, 1] = 7
            assert map[1, 1] == 7
            assert map.y[3] == 7
        map.reset()

    def test_ylm_multiple_l_to_scalar(self, map):
        if map._spectral:
            map[:, 1, 0] = 7
            assert np.all(map[:, 1, 0] == 7)
            inds = [l ** 2 + l + 1 for l in range(1, map.ydeg + 1)] 
            assert np.all(map.y[inds, 0] == 7)
        elif map._temporal:
            map[:, 1, 0] = 7
            assert np.all(map[:, 1, 0] == 7)
            inds = [l ** 2 + l + 1 for l in range(1, map.ydeg + 1)] 
            assert np.all(map.y.reshape(map.nt, -1).transpose()[inds, 0] == 7)
        else:
            map[:, 1] = 7
            assert np.all(map[:, 1] == 7)
            inds = [l ** 2 + l + 1 for l in range(1, map.ydeg + 1)] 
            assert np.all(map.y[inds] == 7)
        map.reset()

    def test_ylm_multiple_m_to_scalar(self, map):
        if map._spectral:
            map[1, :, 0] = 7
            assert np.all(map[1, :, 0] == 7)
            inds = [1, 2, 3]
            assert np.all(map.y[inds, 0] == 7)
        elif map._temporal:
            map[1, :, 0] = 7
            assert np.all(map[1, :, 0] == 7)
            inds = [1, 2, 3]
            assert np.all(map.y.reshape(map.nt, -1).transpose()[inds, 0] == 7)
        else:
            map[1, :] = 7
            assert np.all(map[1, :] == 7)
            inds = [1, 2, 3]
            assert np.all(map.y[inds] == 7)
        map.reset()

    def test_ylm_multiple_x_to_scalar(self, map):
        if map._spectral:
            map[1, 1, :] = 7
            assert np.all(map[1, 1, :] == 7)
            assert np.all(map.y[3, :] == 7)
        elif map._temporal:
            map[1, 1, :] = 7
            assert np.all(map[1, 1, :] == 7)
            assert np.all(map.y.reshape(map.nt, -1).transpose()[3, :] == 7)
        else:
            pass
        map.reset()

    def test_ylm_multiple_lm_to_scalar(self, map):
        if map._spectral:
            map[1:, :, 0] = 7
            assert np.all(map[1:, :, 0] == 7)
            assert np.all(map.y[1:, 0] == 7)
        elif map._temporal:
            map[1:, :, 0] = 7
            assert np.all(map[1:, :, 0] == 7)
            assert np.all(map.y.reshape(map.nt, -1).transpose()[1:, 0] == 7)
        else:
            map[1:, :] = 7
            assert np.all(map[1:, :] == 7)
            assert np.all(map.y[1:] == 7)
        map.reset()

    def test_ylm_multiple_lx_to_scalar(self, map):
        if map._spectral:
            map[1:, 1, :] = 7
            assert np.all(map[1:, 1, :] == 7)
            inds = [l ** 2 + l + 1 for l in range(1, map.ydeg + 1)] 
            assert np.all(map.y[inds, :] == 7)
        elif map._temporal:
            map[1:, 1, :] = 7
            assert np.all(map[1:, 1, :] == 7)
            inds = [l ** 2 + l + 1 for l in range(1, map.ydeg + 1)] 
            assert np.all(map.y.reshape(map.nt, -1).transpose()[inds, :] == 7)
        else:
            pass
        map.reset()

    def test_ylm_multiple_mx_to_scalar(self, map):
        if map._spectral:
            map[1, :, :] = 7
            assert np.all(map[1, :, :] == 7)
            inds = [1, 2, 3]
            assert np.all(map.y[inds, :] == 7)
        elif map._temporal:
            map[1, :, :] = 7
            assert np.all(map[1, :, :] == 7)
            inds = [1, 2, 3]
            assert np.all(map.y.reshape(map.nt, -1).transpose()[inds, :] == 7)
        else:
            pass
        map.reset()

    def test_ylm_multiple_lmx_to_scalar(self, map):
        if map._spectral:
            map[1:, :, :] = 7
            assert np.all(map[1:, :, :] == 7)
            assert np.all(map.y[1:, :] == 7)
        elif map._temporal:
            map[1:, :, :] = 7
            assert np.all(map[1:, :, :] == 7)
            assert np.all(map.y.reshape(map.nt, -1).transpose()[1:, :] == 7)
        else:
            pass
        map.reset()

    def test_ylm_multiple_l_to_vector(self, map):
        if map._spectral:
            map[:, 1, 0] = range(1, map.ydeg + 1)
            assert np.allclose(map[:, 1, 0].flatten(), range(1, map.ydeg + 1))
            inds = [l ** 2 + l + 1 for l in range(1, map.ydeg + 1)] 
            assert np.allclose(map.y[inds, 0], range(1, map.ydeg + 1))
        elif map._temporal:
            map[:, 1, 0] = range(1, map.ydeg + 1)
            assert np.allclose(map[:, 1, 0].flatten(), range(1, map.ydeg + 1))
            inds = [l ** 2 + l + 1 for l in range(1, map.ydeg + 1)] 
            assert np.allclose(map.y.reshape(map.nt, -1).transpose()[inds, 0], range(1, map.ydeg + 1))
        else:
            map[:, 1] = range(1, map.ydeg + 1)
            assert np.allclose(map[:, 1], range(1, map.ydeg + 1))
            inds = [l ** 2 + l + 1 for l in range(1, map.ydeg + 1)] 
            assert np.allclose(map.y[inds], range(1, map.ydeg + 1))
        map.reset()

    def test_ylm_multiple_m_to_vector(self, map):
        if map._spectral:
            map[1, :, 0] = [1, 2, 3]
            inds = [1, 2, 3]
            assert np.allclose(map[1, :, 0].flatten(), inds)
            assert np.allclose(map.y[inds, 0], inds)
        elif map._temporal:
            map[1, :, 0] = [1, 2, 3]
            inds = [1, 2, 3]
            assert np.allclose(map[1, :, 0].flatten(), inds)
            assert np.allclose(map.y.reshape(map.nt, -1).transpose()[inds, 0], inds)
        else:
            map[1, :] = [1, 2, 3]
            inds = [1, 2, 3]
            assert np.allclose(map[1, :], inds)
            assert np.allclose(map.y[inds], inds)
        map.reset()

    def test_ylm_multiple_w_to_vector(self, map):
        if map._spectral:
            map[1, 1, :] = [1, 2, 3]
            assert np.allclose(map[1, 1, :], [1, 2, 3])
            assert np.allclose(map.y[3, :], [1, 2, 3])
        elif map._temporal:
            map[1, 1, :] = [1, 2, 3]
            assert np.allclose(map[1, 1, :], [1, 2, 3])
            assert np.allclose(map.y.reshape(map.nt, -1).transpose()[3, :], [1, 2, 3])
        else:
            pass
        map.reset()

    def test_ylm_multiple_lm_to_vector(self, map):
        if map._spectral:
            map[1:, :, 0] = np.arange(1, map.Ny)
            assert np.allclose(map[1:, :, 0].flatten(), np.arange(1, map.Ny))
            assert np.allclose(map.y[1:, 0], np.arange(1, map.Ny))
        elif map._temporal:
            map[1:, :, 0] = np.arange(1, map.Ny)
            assert np.allclose(map[1:, :, 0].flatten(), np.arange(1, map.Ny))
            assert np.allclose(map.y.reshape(map.nt, -1).transpose()[1:, 0], np.arange(1, map.Ny))
        else:
            map[1:, :] = np.arange(1, map.Ny)
            assert np.allclose(map[1:, :], np.arange(1, map.Ny))
            assert np.allclose(map.y[1:], np.arange(1, map.Ny))
        map.reset()

    def test_ylm_multiple_lx_to_vector(self, map):
        if map._spectral:
            vals = np.array([np.arange(1, map.ydeg + 1) + i 
                             for i in range(map.nw)]).transpose()
            map[1:, 1, :] = vals
            assert np.allclose(map[1:, 1, :], vals)
            inds = [l ** 2 + l + 1 for l in range(1, map.ydeg + 1)]
            assert np.allclose(map.y[inds, :], vals)
        elif map._temporal:
            vals = np.array([np.arange(1, map.ydeg + 1) + i 
                             for i in range(map.nt)]).transpose()
            map[1:, 1, :] = vals
            assert np.allclose(map[1:, 1, :], vals)
            inds = [l ** 2 + l + 1 for l in range(1, map.ydeg + 1)]
            assert np.allclose(map.y.reshape(map.nt, -1).transpose()[inds, :], vals)
        else:
            pass
        map.reset()

    def test_ylm_multiple_mx_to_vector(self, map):
        if map._spectral:
            vals = np.array([[1 + i, 2 + i, 3 + i] 
                             for i in range(map.nw)]).transpose()
            map[1, :, :] = vals
            assert np.allclose(map[1, :, :], vals)
            inds = [1, 2, 3]
            assert np.allclose(map.y[inds, :], vals)
        elif map._temporal:
            vals = np.array([[1 + i, 2 + i, 3 + i] 
                             for i in range(map.nt)]).transpose()
            map[1, :, :] = vals
            assert np.allclose(map[1, :, :], vals)
            inds = [1, 2, 3]
            assert np.allclose(map.y.reshape(map.nt, -1).transpose()[inds, :], vals)
        else:
            pass
        map.reset()

    def test_ylm_multiple_lmx_to_vector(self, map):
        if map._spectral:
            np.random.seed(43)
            vals = np.random.randn(map.Ny - 1, map.nw)
            map[1:, :, :] = vals
            assert np.allclose(map[1:, :, :], vals)
            assert np.allclose(map.y[1:, :], vals)
        elif map._temporal:
            np.random.seed(43)
            vals = np.random.randn(map.Ny - 1, map.nt)
            map[1:, :, :] = vals
            assert np.allclose(map[1:, :, :], vals)
            assert np.allclose(map.y.reshape(map.nt, -1).transpose()[1:, :], vals)
        else:
            pass
        map.reset()

    def test_ul_single_to_scalar(self, map):
        map[1] = 7
        assert map[1] == 7
        assert map.u[1] == 7
        map.reset()

    def test_ul_multiple_to_scalar(self, map):
        map[1:] = 7
        assert np.all(map[1:] == 7)
        assert np.all(map.u[1:] == 7)
        map.reset()

    def test_ul_multiple_to_vector(self, map):
        map[1:] = [1, 2]
        assert np.allclose(map[1:], [1, 2])
        assert np.allclose(map.u[1:], [1, 2])
        map.reset()