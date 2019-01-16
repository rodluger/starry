"""Test coefficient getters/setters."""
import starry2
import numpy as np

# Instatiate all the map types
lmax = 3
map_default_double = starry2.Map(lmax)
map_default_multi = starry2.Map(lmax, multi=True)
map_spectral_double = starry2.Map(lmax, nw=3)
map_spectral_multi = starry2.Map(lmax, nw=3, multi=True)
map_temporal_double = starry2.Map(lmax, nt=3)
map_temporal_multi = starry2.Map(lmax, nt=3, multi=True)


def test_default_flags():
    assert map_default_double.y.flags.writeable == False
    assert map_default_double[:, :].flags.writeable == False
    assert map_default_double.u.flags.writeable == False
    assert map_default_double[:].flags.writeable == False


def test_default_ylm_single():
    for map in (map_default_double, map_default_multi):
        map.reset()
        map[0, 0] = 1
        assert map.y[0] == 1


def test_default_ylm_l_slice():
    for map in (map_default_double, map_default_multi):
        map.reset()
        map[:, 0] = 1
        assert np.allclose(map[:, 0], 1)
        assert np.all([map[l, 0] == 1 for l in range(lmax + 1)])


def test_default_ylm_m_slice():
    for map in (map_default_double, map_default_multi):
        map.reset()
        map[lmax, :] = 1
        assert np.allclose(map[lmax, :], 1)
        assert np.all([map[lmax, m] == 1 for m in range(-lmax, lmax + 1)])


def test_default_ylm_lm_slice():
    for map in (map_default_double, map_default_multi):
        map.reset()
        map[:, :] = 1
        assert(np.allclose(map[:, :], 1))
        assert np.allclose(map.y, 1)


def test_default_ylm_vector_slice():
    for map in (map_default_double, map_default_multi):
        map.reset()
        yl0 = np.arange(lmax + 1)
        map[:, 0] = yl0
        assert np.allclose(map[:, 0], yl0)
        assert np.all([map[l, 0] == yl0[l] for l in range(lmax + 1)])


def test_default_ul_single():
    for map in (map_default_double, map_default_multi):
        map.reset()
        map[1] = 1
        assert map.u[0] == 1


def test_default_ul_l_slice():
    for map in (map_default_double, map_default_multi):
        map.reset()
        map[:] = 1
        assert np.allclose(map[:], 1)
        assert np.all([map[l] == 1 for l in range(1, lmax + 1)])


def test_default_ul_vector_slice():
    for map in (map_default_double, map_default_multi):
        map.reset()
        ul = np.arange(lmax)
        map[:] = ul
        assert np.allclose(map[:], ul)
        assert np.all([map[l] == ul[l - 1] for l in range(1, lmax + 1)])


def test_spectral_flags():
    assert map_spectral_double.y.flags.writeable == False
    assert map_spectral_double[:, :].flags.writeable == False
    assert map_spectral_double[:, :][0].flags.writeable == False
    assert map_spectral_double.u.flags.writeable == False
    assert map_spectral_double[:].flags.writeable == False
    assert map_spectral_double[:][0].flags.writeable == False


def test_spectral_ylm_single():
    for map in (map_spectral_double, map_spectral_multi):
        map.reset()
        vec = np.arange(map.nw)
        map[0, 0] = vec
        assert np.allclose(map.y[0], vec)


def test_spectral_ylm_l_slice():
    for map in (map_spectral_double, map_spectral_multi):
        map.reset()
        vec = np.arange(map.nw)
        map[:, 0] = vec
        assert np.allclose(map[:, 0], vec)
        assert np.all([np.allclose(map[l, 0], vec) for l in range(lmax + 1)])


def test_spectral_ylm_m_slice():
    for map in (map_spectral_double, map_spectral_multi):
        map.reset()
        vec = np.arange(map.nw)
        map[lmax, :] = vec
        assert np.allclose(map[lmax, :], vec)
        assert np.all([np.allclose(map[lmax, m], vec) for m in range(-lmax, lmax + 1)])


def test_spectral_ylm_lm_slice():
    for map in (map_spectral_double, map_spectral_multi):
        map.reset()
        vec = np.arange(map.nw)
        map[:, :] = vec
        assert(np.allclose(map[:, :], vec))
        assert np.allclose(map.y, vec)


def test_spectral_ylm_vector_slice():
    for map in (map_spectral_double, map_spectral_multi):
        map.reset()
        vec = np.arange(map.nw * (lmax + 1)).reshape(lmax + 1, map.nw)
        map[:, 0] = vec
        assert(np.allclose(map[:, 0], vec))
        assert np.all([np.allclose(map[l, 0], vec[l]) for l in range(lmax + 1)])


def test_spectral_ul_single():
    for map in (map_spectral_double, map_spectral_multi):
        map.reset()
        vec = np.arange(map.nw)
        map[1] = vec
        assert np.allclose(map.u[0], vec)


def test_spectral_ul_l_slice():
    for map in (map_spectral_double, map_spectral_multi):
        map.reset()
        vec = np.arange(map.nw)
        map[:] = vec
        assert np.allclose(map[:], vec)
        assert np.all([np.allclose(map[l], vec) for l in range(1, lmax + 1)])


def test_spectral_ul_vector_slice():
    for map in (map_spectral_double, map_spectral_multi):
        map.reset()
        vec = np.arange(map.nw * (lmax)).reshape(lmax, map.nw)
        map[:] = vec
        assert np.allclose(map[:], vec)
        assert np.all([np.allclose(map[l], vec[l - 1]) for l in range(1, lmax + 1)])


def test_temporal_flags():
    assert map_temporal_double.y.flags.writeable == False
    assert map_temporal_double[:, :].flags.writeable == False
    assert map_temporal_double[:, :][0].flags.writeable == False
    assert map_temporal_double.u.flags.writeable == False
    assert map_temporal_double[:].flags.writeable == False


def test_temporal_ylm_single():
    for map in (map_temporal_double, map_temporal_multi):
        map.reset()
        vec = np.arange(map.nt)
        map[0, 0] = vec
        assert np.allclose(map.y[0], vec)


def test_temporal_ylm_l_slice():
    for map in (map_temporal_double, map_temporal_multi):
        map.reset()
        vec = np.arange(map.nt)
        map[:, 0] = vec
        assert np.allclose(map[:, 0], vec)
        assert np.all([np.allclose(map[l, 0], vec) for l in range(lmax + 1)])


def test_temporal_ylm_m_slice():
    for map in (map_temporal_double, map_temporal_multi):
        map.reset()
        vec = np.arange(map.nt)
        map[lmax, :] = vec
        assert np.allclose(map[lmax, :], vec)
        assert np.all([np.allclose(map[lmax, m], vec) for m in range(-lmax, lmax + 1)])


def test_temporal_ylm_lm_slice():
    for map in (map_temporal_double, map_temporal_multi):
        map.reset()
        vec = np.arange(map.nt)
        map[:, :] = vec
        assert(np.allclose(map[:, :], vec))
        assert np.allclose(map.y, vec)


def test_temporal_ylm_vector_slice():
    for map in (map_temporal_double, map_temporal_multi):
        map.reset()
        vec = np.arange(map.nt * (lmax + 1)).reshape(lmax + 1, map.nt)
        map[:, 0] = vec
        assert(np.allclose(map[:, 0], vec))
        assert np.all([np.allclose(map[l, 0], vec[l]) for l in range(lmax + 1)])


def test_temporal_ul_single():
    for map in (map_temporal_double, map_temporal_multi):
        map.reset()
        map[1] = 1
        assert map.u[0] == 1


def test_temporal_ul_l_slice():
    for map in (map_temporal_double, map_temporal_multi):
        map.reset()
        map[:] = 1
        assert np.allclose(map[:], 1)
        assert np.all([map[l] == 1 for l in range(1, lmax + 1)])


def test_temporal_ul_vector_slice():
    for map in (map_temporal_double, map_temporal_multi):
        map.reset()
        ul = np.arange(lmax)
        map[:] = ul
        assert np.allclose(map[:], ul)
        assert np.all([map[l] == ul[l - 1] for l in range(1, lmax + 1)])