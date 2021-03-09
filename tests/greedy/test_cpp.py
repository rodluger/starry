# -*- coding: utf-8 -*-
"""
Tests for C++ functions.

"""
from starry import _c_ops as Ops
import numpy as np
from mpmath import ellipe
import pytest


# Only run tests if `STARRY_UNIT_TESTS=1` on compile
cpp = pytest.mark.skipif(
    not Ops.STARRY_UNIT_TESTS, reason="c++ unit tests not found"
)


@cpp
def test_E():
    """
    Incomplete elliptic integral of the second kind

        E(k^2, phi)

    used in oblate starry.

    This function is specially coded to allow negative values of
    the parameter k^2. It is defined only for

        -3pi/2 <= phi <= pi/2

    which is the domain required by oblate starry.

    """
    angle1s = np.linspace(0, 2 * np.pi, 10)
    angle2s = np.linspace(0, 2 * np.pi, 100)
    k2s = [-100, -5.0, -1.5, -1.0, -0.5, 0.0, 1.0, 0.5, 1.5, 5.0, 100]
    for k2 in k2s:
        for angle1 in angle1s:
            for angle2 in angle2s:

                # This is the angle 2 * u in the oblate integrals
                phi1 = 2 * (np.pi - 2 * angle1) / 4
                phi2 = 2 * (np.pi - 2 * angle2) / 4

                # Compute using starry
                E_starry, E_starry_derivs = Ops.E(k2, np.array([phi1, phi2]))

                # Compute using mpmath
                E_mpmath = float((ellipe(phi2, k2) - ellipe(phi1, k2)).real)

                # Check
                assert np.allclose(
                    E_starry, E_mpmath
                ), "k2={:0.1f}, angle1={:0.2f}, angle2={:0.2f}: {:0.3f} != {:0.3f}".format(
                    k2, angle1, angle2, E_starry, E_mpmath
                )
