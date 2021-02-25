# -*- coding: utf-8 -*-
"""
Test map index setting.

"""
import starry
import numpy as np
import pytest


@pytest.fixture(scope="class", params=[(None,), (3,)])
def map(request):
    (nw,) = request.param
    map = starry.Map(ydeg=5, udeg=2, nw=nw)
    return map


class TestGettersAndSetters:
    """Test the Map getters and setters."""

    def test_ylm_single_to_scalar(self, map):
        map.reset()
        if map.nw:
            map[1, 1, 0] = 7
            assert map[1, 1, 0] == 7
            assert map.y[3, 0] == 7
        else:
            map[1, 1] = 7
            assert map[1, 1] == 7
            assert map.y[3] == 7

    def test_ylm_multiple_l_to_scalar(self, map):
        map.reset()
        if map.nw:
            map[1:, 1, 0] = 7
            assert np.all(map[:, 1, 0] == 7)
            inds = [l ** 2 + l + 1 for l in range(1, map.ydeg + 1)]
            assert np.all(map.y[inds, 0] == 7)
        else:
            map[1:, 1] = 7
            assert np.all(map[:, 1] == 7)
            inds = [l ** 2 + l + 1 for l in range(1, map.ydeg + 1)]
            assert np.all(map.y[inds] == 7)

    def test_ylm_multiple_m_to_scalar(self, map):
        map.reset()
        if map.nw:
            map[1, :, 0] = 7
            assert np.all(map[1, :, 0] == 7)
            inds = [1, 2, 3]
            assert np.all(map.y[inds, 0] == 7)
        else:
            map[1, :] = 7
            assert np.all(map[1, :] == 7)
            inds = [1, 2, 3]
            assert np.all(map.y[inds] == 7)

    def test_ylm_multiple_x_to_scalar(self, map):
        map.reset()
        if map.nw:
            map[1, 1, :] = 7
            assert np.all(map[1, 1, :] == 7)
            assert np.all(map.y[3, :] == 7)
        else:
            pass

    def test_ylm_multiple_lm_to_scalar(self, map):
        map.reset()
        if map.nw:
            map[1:, :, 0] = 7
            assert np.all(map[1:, :, 0] == 7)
            assert np.all(map.y[1:, 0] == 7)
        else:
            map[1:, :] = 7
            assert np.all(map[1:, :] == 7)
            assert np.all(map.y[1:] == 7)

    def test_ylm_multiple_lx_to_scalar(self, map):
        map.reset()
        if map.nw:
            map[1:, 1, :] = 7
            assert np.all(map[1:, 1, :] == 7)
            inds = [l ** 2 + l + 1 for l in range(1, map.ydeg + 1)]
            assert np.all(map.y[inds, :] == 7)
        else:
            pass

    def test_ylm_multiple_mx_to_scalar(self, map):
        map.reset()
        if map.nw:
            map[1, :, :] = 7
            assert np.all(map[1, :, :] == 7)
            inds = [1, 2, 3]
            assert np.all(map.y[inds, :] == 7)
        else:
            pass

    def test_ylm_multiple_lmx_to_scalar(self, map):
        map.reset()
        if map.nw:
            map[1:, :, :] = 7
            assert np.all(map[1:, :, :] == 7)
            assert np.all(map.y[1:, :] == 7)
        else:
            pass

    def test_ylm_multiple_l_to_vector(self, map):
        map.reset()
        if map.nw:
            map[1:, 1, 0] = np.atleast_2d(np.arange(1, map.ydeg + 1)).T
            assert np.allclose(map[:, 1, 0].flatten(), range(1, map.ydeg + 1))
            inds = [l ** 2 + l + 1 for l in range(1, map.ydeg + 1)]
            assert np.allclose(map.y[inds, 0], range(1, map.ydeg + 1))
        else:
            map[1:, 1] = range(1, map.ydeg + 1)
            assert np.allclose(map[:, 1], range(1, map.ydeg + 1))
            inds = [l ** 2 + l + 1 for l in range(1, map.ydeg + 1)]
            assert np.allclose(map.y[inds], range(1, map.ydeg + 1))

    def test_ylm_multiple_m_to_vector(self, map):
        map.reset()
        if map.nw:
            map[1, :, 0] = np.array([[1], [2], [3]])
            inds = [1, 2, 3]
            assert np.allclose(map[1, :, 0].flatten(), inds)
            assert np.allclose(map.y[inds, 0], inds)
        else:
            map[1, :] = [1, 2, 3]
            inds = [1, 2, 3]
            assert np.allclose(map[1, :], inds)
            assert np.allclose(map.y[inds], inds)

    def test_ylm_multiple_w_to_vector(self, map):
        map.reset()
        if map.nw:
            map[1, 1, :] = [1, 2, 3]
            assert np.allclose(map[1, 1, :], [1, 2, 3])
            assert np.allclose(map.y[3, :], [1, 2, 3])
        else:
            pass

    def test_ylm_multiple_lm_to_vector(self, map):
        map.reset()
        if map.nw:
            map[1:, :, 0] = np.atleast_2d(np.arange(1, map.Ny)).T
            assert np.allclose(map[1:, :, 0].flatten(), np.arange(1, map.Ny))
            assert np.allclose(map.y[1:, 0], np.arange(1, map.Ny))
        else:
            map[1:, :] = np.arange(1, map.Ny)
            assert np.allclose(map[1:, :], np.arange(1, map.Ny))
            assert np.allclose(map.y[1:], np.arange(1, map.Ny))

    def test_ylm_multiple_lx_to_vector(self, map):
        map.reset()
        if map.nw:
            vals = np.array(
                [np.arange(1, map.ydeg + 1) + i for i in range(map.nw)]
            ).transpose()
            map[1:, 1, :] = vals
            assert np.allclose(map[1:, 1, :], vals)
            inds = [l ** 2 + l + 1 for l in range(1, map.ydeg + 1)]
            assert np.allclose(map.y[inds, :], vals)
        else:
            pass

    def test_ylm_multiple_mx_to_vector(self, map):
        map.reset()
        if map.nw:
            vals = np.array(
                [[1 + i, 2 + i, 3 + i] for i in range(map.nw)]
            ).transpose()
            map[1, :, :] = vals
            assert np.allclose(map[1, :, :], vals)
            inds = [1, 2, 3]
            assert np.allclose(map.y[inds, :], vals)
        else:
            pass

    def test_ylm_multiple_lmx_to_vector(self, map):
        map.reset()
        if map.nw:
            np.random.seed(43)
            vals = np.random.randn(map.Ny - 1, map.nw)
            map[1:, :, :] = vals
            assert np.allclose(map[1:, :, :], vals)
            assert np.allclose(map.y[1:, :], vals)
        else:
            pass

    def test_ul_single_to_scalar(self, map):
        map.reset()
        map[1] = 7
        assert map[1] == 7
        assert map.u[1] == 7

    def test_ul_multiple_to_scalar(self, map):
        map.reset()
        map[1:] = 7
        assert np.all(map[1:] == 7)
        assert np.all(map.u[1:] == 7)

    def test_ul_multiple_to_vector(self, map):
        map.reset()
        map[1:] = [1, 2]
        assert np.allclose(map[1:], [1, 2])
        assert np.allclose(map.u[1:], [1, 2])

    def test_yl_set_constant_coeff(self, map):
        map.reset()
        if map.nw:
            try:
                map[0, 0, :] = 2.0
                raise Exception("")
            except ValueError:
                pass
        else:
            try:
                map[0, 0] = 2.0
                raise Exception("")
            except ValueError:
                pass

    def test_ul_set_constant_coeff(self, map):
        map.reset()
        try:
            map[0] = 0.0
            raise Exception("")
        except ValueError:
            pass

    def test_ul_constant_coeff(self, map):
        map.reset()
        assert np.allclose(map[:], [-1, 0, 0])

    def test_set_all(self, map):
        map.reset()
        if not map.nw:
            map[:, :] = 1
            assert np.allclose(map[:, :], np.ones(map.Ny))

    def test_set_transposed(self, map):
        map.reset()
        if map.nw:
            vals = np.ones((map.nw, map.Ny - 1))
            map[1:, :, :] = vals
            assert np.allclose(map[1:, :, :], vals.T)
