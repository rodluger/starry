# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np
import starry
from theano.tests import unittest_tools


def verify_grad(func, abs_tol=1e-5, rel_tol=1e-5, **kwargs):
    """
    Wrapper for ``theano.tests.unittest_tools.verify_grad``
    that accepts keyword arguments.

    """
    keys = []
    vals = []
    for key, val in kwargs.items():
        keys.append(key)
        vals.append(val)

    def run(*args):
        for key, arg in zip(keys, args):
            kwargs[key] = arg
        return func(**kwargs)

    unittest_tools.verify_grad(run, vals, abs_tol=abs_tol, rel_tol=rel_tol)


def test_doppler(abs_tol=1e-5, rel_tol=1e-5):
    map = starry.Map(ydeg=1, udeg=2, doppler=True)
    kwargs = {
        "y":        np.array([0.25, 0.25, 0.25]),
        "u":        np.array([0.4, 0.26]),
        "theta":    30.0,
        "inc":      75.0,
        "obl":      30.0,
        "alpha":    0.40,
        "veq":      0.40,
        "xo":       0.15,
        "yo":       0.2,
        "zo":       1.0,
        "ro":       0.1
    }
    verify_grad(map.rv, abs_tol=abs_tol, rel_tol=rel_tol, **kwargs)


def test_ylm(abs_tol=1e-5, rel_tol=1e-5):
    map = starry.Map(ydeg=1, udeg=2)
    kwargs = {
        "y":        np.array([0.25, 0.25, 0.25]),
        "u":        np.array([0.4, 0.26]),
        "theta":    30.0,
        "inc":      75.0,
        "obl":      30.0,
        "xo":       np.linspace(-1.5, 1.5, 10),
        "yo":       0.2,
        "zo":       1.0,
        "ro":       0.1
    }
    verify_grad(map.flux, abs_tol=abs_tol, rel_tol=rel_tol, **kwargs)


def test_ld(abs_tol=1e-5, rel_tol=1e-5):
    map = starry.Map(udeg=2)
    kwargs = {
        "u":        np.array([0.4, 0.26]),
        "b":        np.linspace(-1.5, 1.5, 10),
        "zo":       1.0,
        "ro":       0.1
    }
    verify_grad(map.flux, abs_tol=abs_tol, rel_tol=rel_tol, **kwargs)


def test_ld_spectral(abs_tol=1e-5, rel_tol=1e-5):
    map = starry.Map(udeg=3, nw=3)
    kwargs = {
        "u":        np.array([[0.8, 0.2, 0.4], 
                              [0.1, 0.13, 0.4], 
                              [0.3, 0.3, 0.3]]),
        "b":        np.linspace(-1.5, 1.5, 10),
        "zo":       1.0,
        "ro":       0.1
    }
    verify_grad(map.flux, abs_tol=abs_tol, rel_tol=rel_tol, **kwargs)


if __name__ == "__main__":
    test_doppler()
    test_ylm()
    test_ld()
    test_ld_spectral()