"""
Test the map evaluation against some benchmarks.

"""
import starry
import numpy as np


def test_default_scalar():
    map = starry.Map(1)
    map[:, :] = 1
    x = 0.3
    y = 0.4
    z = np.sqrt(1 - x ** 2 - y ** 2)
    assert np.allclose(
        map(x=x, y=y), 
        (1.0 + np.sqrt(3) * (y + z + x)) / np.pi
    )


def test_default_vector():
    map = starry.Map(1)
    map[:, :] = 1
    x = np.array([0.3, 0.4])
    y = np.array([0.4, 0.5])
    z = np.sqrt(1 - x ** 2 - y ** 2)
    assert np.allclose(
        map(x=x, y=y), 
        (1.0 + np.sqrt(3) * (y + z + x)) / np.pi
    )


def test_spectral_scalar():
    map = starry.Map(1, nw=2)
    map[1:, :, :] = [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]
    x = 0.3
    y = 0.4
    z = np.sqrt(1 - x ** 2 - y ** 2)
    assert np.allclose(
        map(x=x, y=y),
        np.array([
            (1.0 + 1.0 * np.sqrt(3) * (y + z + x)) / np.pi,
            (1.0 + 2.0 * np.sqrt(3) * (y + z + x)) / np.pi
        ])
    )
    

def test_spectral_vector():
    map = starry.Map(1, nw=2)
    map[1:, :, :] = [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]
    x = np.array([0.3, 0.4])
    y = np.array([0.4, 0.5])
    z = np.sqrt(1 - x ** 2 - y ** 2)
    assert np.allclose(
        map(x=x, y=y), 
        np.array([
            (1.0 + 1.0 * np.sqrt(3) * (y + z + x)) / np.pi,
            (1.0 + 2.0 * np.sqrt(3) * (y + z + x)) / np.pi
        ]).transpose()
    )


def test_temporal_scalar():
    map = starry.Map(1, nt=2)
    map[1:, :, :] = [[1.0, 0.5], [1.0, 0.5], [1.0, 0.5]]
    t = 1.0
    x = 0.3
    y = 0.4
    z = np.sqrt(1 - x ** 2 - y ** 2)
    assert np.allclose(
        map(t=t, x=x, y=y), 
        (1.0 + (1 + 0.5 * t) * np.sqrt(3) * (y + z + x)) / np.pi
    )


def test_temporal_vector():
    map = starry.Map(1, nt=2)
    map[1:, :, :] = [[1.0, 0.5], [1.0, 0.5], [1.0, 0.5]]
    t = np.array([1.0, 2.0])
    x = 0.3
    y = 0.4
    z = np.sqrt(1 - x ** 2 - y ** 2)
    assert np.allclose(
        map(t=t, x=x, y=y), 
        (1.0 + (1 + 0.5 * t) * np.sqrt(3) * (y + z + x)) / np.pi
    )