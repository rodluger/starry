# -*- coding: utf-8 -*-
"""
Tests for C++ functions.

"""
from starry import _c_ops as Ops
import numpy as np
from mpmath import ellipe, ellipf
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
    the parameter k^2. It is defined for all real phi.

    """
    phi1s = np.linspace(-6 * np.pi, 6 * np.pi, 31)
    phi2s = np.linspace(-6 * np.pi, 6 * np.pi, 31)
    k2s = [-100, -5.0, -1.5, -1.0, -0.5, 0.0, 1.0, 0.5, 1.5, 5.0, 100]
    for k2 in k2s:
        for phi1 in phi1s:
            for phi2 in phi2s:

                # Compute using starry
                E_starry, E_starry_derivs = Ops.E(k2, np.array([phi1, phi2]))

                # Compute using mpmath
                E_mpmath = float((ellipe(phi2, k2) - ellipe(phi1, k2)).real)

                # Check
                assert np.allclose(
                    E_starry, E_mpmath
                ), "k2={:0.1f}, phi1={:0.2f}, phi2={:0.2f}: {:0.3f} != {:0.3f}".format(
                    k2, phi1, phi2, E_starry, E_mpmath
                )


@cpp
def test_F():
    """
    Incomplete elliptic integral of the first kind

        F(k^2, phi)

    used in oblate starry.

    This function is specially coded to allow negative values of
    the parameter k^2. It is defined for all real phi.

    """
    phi1s = np.linspace(-6 * np.pi, 6 * np.pi, 31)
    phi2s = np.linspace(-6 * np.pi, 6 * np.pi, 31)
    k2s = [-100, -5.0, -1.5, -1.0, -0.5, 0.0, 1.0, 0.5, 1.5, 5.0, 100]
    for k2 in k2s:
        for phi1 in phi1s:
            for phi2 in phi2s:

                # Compute using starry
                F_starry, F_starry_derivs = Ops.F(k2, np.array([phi1, phi2]))

                # Compute using mpmath
                if k2 == 1.0:
                    # I /think/ mpmath is wrong in this limit!
                    F_mpmath = float(
                        (
                            ellipf(phi2, k2 + 1e-12) - ellipf(phi1, k2 + 1e-12)
                        ).real
                    )
                else:
                    F_mpmath = float(
                        (ellipf(phi2, k2) - ellipf(phi1, k2)).real
                    )

                # Check
                assert np.allclose(
                    F_starry, F_mpmath
                ), "k2={:0.1f}, phi1={:0.2f}, phi2={:0.2f}: {:0.3f} != {:0.3f}".format(
                    k2, phi1, phi2, F_starry, F_mpmath
                )
