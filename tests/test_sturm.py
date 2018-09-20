"""Test Sturm's theorem."""
from starry import Map
import numpy as np
from scipy.optimize import curve_fit


def IofMu(mu, *u):
    """Return the specific intensity as a function of `mu`."""
    return (1 - np.sum([u[l] * (1 - mu) ** l
                        for l in range(1, len(u))], axis=0))


def test_sturm():
    mu = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    I = [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1]
    guess = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    u, _ = curve_fit(IofMu, mu, I, guess)
    map = Map(10)
    map[:] = u[1:]
    assert map.is_physical() is False


if __name__ == "__main__":
    test_sturm()
