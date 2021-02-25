# -*- coding: utf-8 -*-
from .. import config
from .._constants import *
from .ops import (
    sTOp,
    rTReflectedOp,
    sTReflectedOp,
    dotROp,
    tensordotRzOp,
    FOp,
    spotYlmOp,
    pTOp,
    minimizeOp,
    LDPhysicalOp,
    LimbDarkOp,
    GetClOp,
    RaiseValueErrorOp,
    RaiseValueErrorIfOp,
    OrenNayarOp,
)
from .utils import logger, autocompile, is_theano, clear_cache
from .math import lazy_math as math
import theano
import theano.tensor as tt
import theano.sparse as ts
from theano.ifelse import ifelse
from scipy.special import legendre as LegendreP
import numpy as np
from astropy import units
import os
import exoplanet


# C extensions are not installed on RTD
if os.getenv("READTHEDOCS") == "True":  # pragma: no cover
    _c_ops = None
else:
    from .. import _c_ops


__all__ = ["OpsYlm", "OpsLD", "OpsReflected", "OpsRV", "OpsSystem"]


class OpsYlm(object):
    """Class housing Theano operations for spherical harmonics maps."""

    def __init__(self, ydeg, udeg, fdeg, nw, reflected=False, **kwargs):
        # Ingest kwargs
        self.ydeg = ydeg
        self.udeg = udeg
        self.fdeg = fdeg
        self.deg = ydeg + udeg + fdeg
        self.filter = (fdeg > 0) or (udeg > 0)
        self.nw = nw
        self._reflected = reflected
        self.Ny = (self.ydeg + 1) ** 2

        # Instantiate the C++ Ops
        config.rootHandler.terminator = ""
        logger.info("Pre-computing some matrices... ")
        self._c_ops = _c_ops.Ops(ydeg, udeg, fdeg)
        config.rootHandler.terminator = "\n"
        logger.info("Done.")

        # Solution vectors
        self._sT = sTOp(self._c_ops.sT, self._c_ops.N)
        self._rT = tt.shape_padleft(tt.as_tensor_variable(self._c_ops.rT))
        self._rTA1 = tt.shape_padleft(tt.as_tensor_variable(self._c_ops.rTA1))

        # Change of basis matrices
        self._A = ts.as_sparse_variable(self._c_ops.A)
        self._A1 = ts.as_sparse_variable(self._c_ops.A1)
        self._A1Inv = ts.as_sparse_variable(self._c_ops.A1Inv)

        # Rotation operations
        self._tensordotRz = tensordotRzOp(self._c_ops.tensordotRz)
        self._dotR = dotROp(self._c_ops.dotR)

        # Filter
        # TODO: Make the filter operator sparse
        self._F = FOp(self._c_ops.F, self._c_ops.N, self._c_ops.Ny)

        # Misc
        self._spotYlm = spotYlmOp(
            self._c_ops.spotYlm, self.ydeg, self.nw
        )  # Deprecated
        self._pT = pTOp(self._c_ops.pT, self.deg)
        if self.nw is None:
            if self._reflected:
                self._minimize = minimizeOp(
                    self.unweighted_intensity,
                    self.P,
                    self.ydeg,
                    self.udeg,
                    self.fdeg,
                )
            else:
                self._minimize = minimizeOp(
                    self.intensity, self.P, self.ydeg, self.udeg, self.fdeg
                )
        else:
            # TODO: Implement minimization for spectral maps?
            self._minimize = None
        self._LimbDarkIsPhysical = LDPhysicalOp(_c_ops.nroots)

    @property
    def rT(self):
        return self._rT

    @property
    def rTA1(self):
        return self._rTA1

    @property
    def A(self):
        return self._A

    @property
    def A1(self):
        return self._A1

    @property
    def A1Inv(self):
        return self._A1Inv

    @autocompile
    def sT(self, b, r):
        return self._sT(b, r)

    @autocompile
    def tensordotRz(self, matrix, theta):
        return self._tensordotRz(matrix, theta)

    @autocompile
    def dotR(self, matrix, ux, uy, uz, theta):
        return self._dotR(matrix, ux, uy, uz, theta)

    @autocompile
    def F(self, u, f):
        return self._F(u, f)

    @autocompile
    def spotYlm(self, amp, sigma, lat, lon):
        # Deprecated
        return self._spotYlm(amp, sigma, lat, lon)

    @autocompile
    def pT(self, x, y, z):
        return self._pT(x, y, z)

    @autocompile
    def limbdark_is_physical(self, u):
        """Return True if the limb darkening profile is physical."""
        return self._LimbDarkIsPhysical(u)

    @autocompile
    def get_minimum(self, y):
        """Compute the location and value of the intensity minimum."""
        return self._minimize(y)

    @autocompile
    def X(self, theta, xo, yo, zo, ro, inc, obl, u, f):
        """Compute the light curve design matrix."""
        # Determine shapes
        rows = theta.shape[0]
        cols = self.rTA1.shape[1]
        X = tt.zeros((rows, cols))

        # Compute the occultation mask
        b = tt.sqrt(xo ** 2 + yo ** 2)
        b_rot = tt.ge(b, 1.0 + ro) | tt.le(zo, 0.0) | tt.eq(ro, 0.0)
        b_occ = tt.invert(b_rot)
        i_rot = tt.arange(b.size)[b_rot]
        i_occ = tt.arange(b.size)[b_occ]

        # Compute filter operator
        if self.filter:
            F = self.F(u, f)

        # Rotation operator
        if self.filter:
            rTA1 = ts.dot(tt.dot(self.rT, F), self.A1)
        else:
            rTA1 = self.rTA1
        rTA1 = tt.tile(rTA1, (theta[i_rot].shape[0], 1))
        X = tt.set_subtensor(
            X[i_rot], self.right_project(rTA1, inc, obl, theta[i_rot])
        )

        # Occultation + rotation operator
        sT = self.sT(b[i_occ], ro)
        sTA = ts.dot(sT, self.A)
        theta_z = tt.arctan2(xo[i_occ], yo[i_occ])
        sTAR = self.tensordotRz(sTA, theta_z)
        if self.filter:
            A1InvFA1 = ts.dot(ts.dot(self.A1Inv, F), self.A1)
            sTAR = tt.dot(sTAR, A1InvFA1)
        X = tt.set_subtensor(
            X[i_occ], self.right_project(sTAR, inc, obl, theta[i_occ])
        )

        return X

    @autocompile
    def flux(self, theta, xo, yo, zo, ro, inc, obl, y, u, f):
        """Compute the light curve."""
        return tt.dot(self.X(theta, xo, yo, zo, ro, inc, obl, u, f), y)

    @autocompile
    def P(self, lat, lon):
        """Compute the pixelization matrix, no filters or illumination."""
        # Get the Cartesian points
        xpt, ypt, zpt = self.latlon_to_xyz(lat, lon)

        # Compute the polynomial basis at the point
        pT = self.pT(xpt, ypt, zpt)[:, : (self.ydeg + 1) ** 2]

        # Transform to the Ylm basis
        pTA1 = ts.dot(pT, self.A1)

        # NOTE: The factor of `pi` ensures the correct normalization.
        # This is *different* from the derivation in the paper, but it's
        # due to the fact that the in starry we normalize the spherical
        # harmonics in a slightly strange way (they're normalized so that
        # the integral of Y_{0,0} over the unit sphere is 4, not 4pi).
        # This is useful for thermal light maps, where the flux from a map
        # with Y_{0,0} = 1 is *unity*. But it messes up things for reflected
        # light maps, so we need to account for that here.
        if self._reflected:
            pTA1 *= np.pi

        # We're done
        return pTA1

    @autocompile
    def intensity(self, lat, lon, y, u, f, theta, ld):
        """Compute the intensity at a point or a set of points."""
        # Get the Cartesian points
        xpt, ypt, zpt = self.latlon_to_xyz(lat, lon)

        # Compute the polynomial basis at the point
        pT = self.pT(xpt, ypt, zpt)

        # Transform the map to the polynomial basis
        A1y = ts.dot(self.A1, y)

        # Apply the filter
        if self.filter:
            u0 = tt.zeros_like(u)
            u0 = tt.set_subtensor(u0[0], -1.0)
            A1y = ifelse(
                ld, tt.dot(self.F(u, f), A1y), tt.dot(self.F(u0, f), A1y)
            )

        # Dot the polynomial into the basis
        return tt.dot(pT, A1y)

    @autocompile
    def render(self, res, projection, theta, inc, obl, y, u, f):
        """Render the map on a Cartesian grid."""
        # Compute the Cartesian grid
        xyz = ifelse(
            tt.eq(projection, STARRY_RECTANGULAR_PROJECTION),
            self.compute_rect_grid(res)[-1],
            ifelse(
                tt.eq(projection, STARRY_MOLLWEIDE_PROJECTION),
                self.compute_moll_grid(res)[-1],
                self.compute_ortho_grid(res)[-1],
            ),
        )

        # Compute the polynomial basis
        pT = self.pT(xyz[0], xyz[1], xyz[2])

        # If orthographic, rotate the map to the correct frame
        if self.nw is None:
            Ry = ifelse(
                tt.eq(projection, STARRY_ORTHOGRAPHIC_PROJECTION),
                self.left_project(
                    tt.transpose(tt.tile(y, [theta.shape[0], 1])),
                    inc,
                    obl,
                    theta,
                ),
                tt.transpose(tt.tile(y, [theta.shape[0], 1])),
            )
        else:
            Ry = ifelse(
                tt.eq(projection, STARRY_ORTHOGRAPHIC_PROJECTION),
                self.left_project(y, inc, obl, tt.tile(theta[0], self.nw)),
                y,
            )

        # Change basis to polynomials
        A1Ry = ts.dot(self.A1, Ry)

        # Apply the filter *only if orthographic*
        if self.filter:
            f0 = tt.zeros_like(f)
            f0 = tt.set_subtensor(f0[0], np.pi)
            u0 = tt.zeros_like(u)
            u0 = tt.set_subtensor(u0[0], -1.0)
            A1Ry = ifelse(
                tt.eq(projection, STARRY_ORTHOGRAPHIC_PROJECTION),
                tt.dot(self.F(u, f), A1Ry),
                tt.dot(self.F(u0, f0), A1Ry),
            )

        # Dot the polynomial into the basis
        res = tt.reshape(tt.dot(pT, A1Ry), [res, res, -1])

        # We need the shape to be (nframes, npix, npix)
        return res.dimshuffle(2, 0, 1)

    @autocompile
    def expand_spot(self, amp, sigma, lat, lon):
        """Return the spherical harmonic expansion of a Gaussian spot [DEPRECATED]."""
        return self.spotYlm(amp, sigma, lat, lon)

    @autocompile
    def compute_ortho_grid(self, res):
        """Compute the polynomial basis on the plane of the sky."""
        # NOTE: I think there's a bug in Theano related to
        # tt.mgrid; I get different results depending on whether the
        # function is compiled using `theano.function()` or if it
        # is evaluated using `.eval()`. The small perturbation to `res`
        # is a hacky fix that ensures that `y` and `x` are of the
        # correct length in all cases I've tested.
        dx = 2.0 / (res - 0.01)
        y, x = tt.mgrid[-1:1:dx, -1:1:dx]
        z = tt.sqrt(1 - x ** 2 - y ** 2)
        y = tt.set_subtensor(y[tt.isnan(z)], np.nan)
        x = tt.reshape(x, [1, -1])
        y = tt.reshape(y, [1, -1])
        z = tt.reshape(z, [1, -1])
        lat = tt.reshape(0.5 * np.pi - tt.arccos(y), [1, -1])
        lon = tt.reshape(tt.arctan(x / z), [1, -1])
        return tt.concatenate((lat, lon)), tt.concatenate((x, y, z))

    @autocompile
    def compute_ortho_grid_inc_obl(self, res, inc, obl):
        """Compute the polynomial basis on the plane of the sky, accounting
        for the map inclination and obliquity."""
        # See NOTE on tt.mgrid bug in `compute_ortho_grid`
        dx = 2.0 / (res - 0.01)
        y, x = tt.mgrid[-1:1:dx, -1:1:dx]
        z = tt.sqrt(1 - x ** 2 - y ** 2)
        y = tt.set_subtensor(y[tt.isnan(z)], np.nan)
        x = tt.reshape(x, [1, -1])
        y = tt.reshape(y, [1, -1])
        z = tt.reshape(z, [1, -1])
        Robl = self.RAxisAngle(tt.as_tensor_variable([0.0, 0.0, 1.0]), -obl)
        Rinc = self.RAxisAngle(
            tt.as_tensor_variable([tt.cos(obl), tt.sin(obl), 0.0]),
            -(0.5 * np.pi - inc),
        )
        R = tt.dot(Robl, Rinc)
        xyz = tt.dot(R, tt.concatenate((x, y, z)))
        x = tt.reshape(xyz[0], [1, -1])
        y = tt.reshape(xyz[1], [1, -1])
        z = tt.reshape(xyz[2], [1, -1])
        lat = tt.reshape(0.5 * np.pi - tt.arccos(y), [1, -1])
        lon = tt.reshape(tt.arctan2(x, z), [1, -1])
        return tt.concatenate((lat, lon)), tt.concatenate((x, y, z))

    @autocompile
    def compute_rect_grid(self, res):
        """Compute the polynomial basis on a rectangular lat/lon grid."""
        # See NOTE on tt.mgrid bug in `compute_ortho_grid`
        dx = np.pi / (res - 0.01)
        lat, lon = tt.mgrid[
            -np.pi / 2 : np.pi / 2 : dx, -3 * np.pi / 2 : np.pi / 2 : 2 * dx
        ]
        x = tt.reshape(tt.cos(lat) * tt.cos(lon), [1, -1])
        y = tt.reshape(tt.cos(lat) * tt.sin(lon), [1, -1])
        z = tt.reshape(tt.sin(lat), [1, -1])
        R = self.RAxisAngle(tt.as_tensor_variable([1.0, 0.0, 0.0]), -np.pi / 2)
        return (
            tt.concatenate(
                (
                    tt.reshape(lat, [1, -1]),
                    tt.reshape(lon + 0.5 * np.pi, [1, -1]),
                )
            ),
            tt.dot(R, tt.concatenate((x, y, z))),
        )

    @autocompile
    def compute_moll_grid(self, res):
        """Compute the polynomial basis on a Mollweide grid."""
        # See NOTE on tt.mgrid bug in `compute_ortho_grid`
        dx = 2 * np.sqrt(2) / (res - 0.01)
        y, x = tt.mgrid[
            -np.sqrt(2) : np.sqrt(2) : dx,
            -2 * np.sqrt(2) : 2 * np.sqrt(2) : 2 * dx,
        ]

        # Make points off-grid nan
        a = np.sqrt(2)
        b = 2 * np.sqrt(2)
        y = tt.where((y / a) ** 2 + (x / b) ** 2 <= 1, y, np.nan)

        # https://en.wikipedia.org/wiki/Mollweide_projection
        theta = tt.arcsin(y / np.sqrt(2))
        lat = tt.arcsin((2 * theta + tt.sin(2 * theta)) / np.pi)
        lon0 = 3 * np.pi / 2
        lon = lon0 + np.pi * x / (2 * np.sqrt(2) * tt.cos(theta))

        # Back to Cartesian, this time on the *sky*
        x = tt.reshape(tt.cos(lat) * tt.cos(lon), [1, -1])
        y = tt.reshape(tt.cos(lat) * tt.sin(lon), [1, -1])
        z = tt.reshape(tt.sin(lat), [1, -1])
        R = self.RAxisAngle(tt.as_tensor_variable([1.0, 0.0, 0.0]), -np.pi / 2)
        return (
            tt.concatenate(
                (
                    tt.reshape(lat, (1, -1)),
                    tt.reshape(lon - 1.5 * np.pi, (1, -1)),
                )
            ),
            tt.dot(R, tt.concatenate((x, y, z))),
        )

    @autocompile
    def right_project(self, M, inc, obl, theta):
        r"""Apply the projection operator on the right.

        Specifically, this method returns the dot product :math:`M \cdot R`,
        where ``M`` is an input matrix and ``R`` is the Wigner rotation matrix
        that transforms a spherical harmonic coefficient vector in the
        input frame to a vector in the observer's frame.
        """
        # Trivial case
        if self.ydeg == 0:
            return M

        # Rotate to the sky frame
        # TODO: Do this in a single compound rotation
        M = self.dotR(
            self.dotR(
                self.dotR(
                    M,
                    -tt.cos(obl),
                    -tt.sin(obl),
                    math.to_tensor(0.0),
                    -(0.5 * np.pi - inc),
                ),
                math.to_tensor(0.0),
                math.to_tensor(0.0),
                math.to_tensor(1.0),
                obl,
            ),
            math.to_tensor(1.0),
            math.to_tensor(0.0),
            math.to_tensor(0.0),
            -0.5 * np.pi,
        )

        # Rotate to the correct phase
        if theta.ndim > 0:
            M = self.tensordotRz(M, theta)

        else:
            M = self.dotR(
                M,
                math.to_tensor(0.0),
                math.to_tensor(0.0),
                math.to_tensor(1.0),
                theta,
            )

        # Rotate to the polar frame
        M = self.dotR(
            M,
            math.to_tensor(1.0),
            math.to_tensor(0.0),
            math.to_tensor(0.0),
            0.5 * np.pi,
        )

        return M

    @autocompile
    def left_project(self, M, inc, obl, theta):
        r"""Apply the projection operator on the left.

        Specifically, this method returns the dot product :math:`R \cdot M`,
        where ``M`` is an input matrix and ``R`` is the Wigner rotation matrix
        that transforms a spherical harmonic coefficient vector in the
        input frame to a vector in the observer's frame.

        """
        # Trivial case
        if self.ydeg == 0:
            return M

        # Note that here we are using the fact that R . M = (M^T . R^T)^T
        MT = tt.transpose(M)

        # Rotate to the polar frame
        MT = self.dotR(
            MT,
            math.to_tensor(1.0),
            math.to_tensor(0.0),
            math.to_tensor(0.0),
            -0.5 * np.pi,
        )

        # Rotate to the correct phase
        if theta.ndim > 0:
            MT = self.tensordotRz(MT, -theta)
        else:
            MT = self.dotR(
                MT,
                math.to_tensor(0.0),
                math.to_tensor(0.0),
                math.to_tensor(1.0),
                -theta,
            )

        # Rotate to the sky frame
        # TODO: Do this in a single compound rotation
        MT = self.dotR(
            self.dotR(
                self.dotR(
                    MT,
                    math.to_tensor(1.0),
                    math.to_tensor(0.0),
                    math.to_tensor(0.0),
                    0.5 * np.pi,
                ),
                math.to_tensor(0.0),
                math.to_tensor(0.0),
                math.to_tensor(1.0),
                -obl,
            ),
            -tt.cos(obl),
            -tt.sin(obl),
            math.to_tensor(0.0),
            (0.5 * np.pi - inc),
        )

        return tt.transpose(MT)

    @autocompile
    def set_map_vector(self, vector, inds, vals):
        """Set the elements of the theano map coefficient tensor."""
        res = tt.set_subtensor(vector[inds], vals * tt.ones_like(vector[inds]))
        return res

    @autocompile
    def latlon_to_xyz(self, lat, lon):
        """Convert lat-lon points to Cartesian points."""
        if lat.ndim == 0:
            lat = tt.shape_padleft(lat, 1)
        if lon.ndim == 0:
            lon = tt.shape_padleft(lon, 1)
        R1 = self.RAxisAngle([1.0, 0.0, 0.0], -lat)
        R2 = self.RAxisAngle([0.0, 1.0, 0.0], lon)
        R = tt.batched_dot(R2, R1)
        xyz = tt.transpose(tt.dot(R, [0.0, 0.0, 1.0]))
        return xyz[0], xyz[1], xyz[2]

    @autocompile
    def RAxisAngle(self, axis=[0, 1, 0], theta=0):
        """Wigner axis-angle rotation matrix."""

        def compute(axis=[0, 1, 0], theta=0):
            axis = tt.as_tensor_variable(axis)
            axis /= axis.norm(2)
            cost = tt.cos(theta)
            sint = tt.sin(theta)

            return tt.reshape(
                tt.as_tensor_variable(
                    [
                        cost + axis[0] * axis[0] * (1 - cost),
                        axis[0] * axis[1] * (1 - cost) - axis[2] * sint,
                        axis[0] * axis[2] * (1 - cost) + axis[1] * sint,
                        axis[1] * axis[0] * (1 - cost) + axis[2] * sint,
                        cost + axis[1] * axis[1] * (1 - cost),
                        axis[1] * axis[2] * (1 - cost) - axis[0] * sint,
                        axis[2] * axis[0] * (1 - cost) - axis[1] * sint,
                        axis[2] * axis[1] * (1 - cost) + axis[0] * sint,
                        cost + axis[2] * axis[2] * (1 - cost),
                    ]
                ),
                [3, 3],
            )

        # If theta is a vector, this is a tensor!
        if hasattr(theta, "ndim") and theta.ndim > 0:
            fn = lambda theta, axis: compute(axis=axis, theta=theta)
            R, _ = theano.scan(fn=fn, sequences=[theta], non_sequences=[axis])
            return R
        else:
            return compute(axis=axis, theta=theta)

    def _spot_setup(
        self, spot_pts=1000, spot_eps=1e-9, spot_smoothing=None, spot_fac=300
    ):
        # The default smoothing depends on `ydeg`
        if spot_smoothing is None:
            if self.ydeg < 4:
                spot_smoothing = 0.5
            else:
                spot_smoothing = 2.0 / self.ydeg

        # Check the cache
        try:
            if (
                (spot_pts == self._spot_pts)
                and (spot_eps == self._spot_eps)
                and (spot_smoothing == self._spot_smoothing)
                and (spot_fac == self._spot_fac)
            ):
                return
        except AttributeError:
            pass

        # Update the cached values
        self._spot_pts = spot_pts
        self._spot_eps = spot_eps
        self._spot_smoothing = spot_smoothing
        self._spot_fac = spot_fac

        # Clear the compiled function cache
        clear_cache(self, self.spot)

        # Pre-compute the linalg stuff
        theta = np.linspace(0, np.pi, spot_pts)
        cost = np.cos(theta)
        B = np.hstack(
            [
                np.sqrt(2 * l + 1) * LegendreP(l)(cost).reshape(-1, 1)
                for l in range(self.ydeg + 1)
            ]
        )
        A = np.linalg.solve(B.T @ B + spot_eps * np.eye(self.ydeg + 1), B.T)
        l = np.arange(self.ydeg + 1)
        i = l * (l + 1)
        S = np.exp(-0.5 * i * spot_smoothing ** 2)
        self._spot_Bp = S[:, None] * A
        self._spot_idx = i
        self._spot_theta = theta
        self._spot_fac = spot_fac

    @autocompile
    def spot(self, contrast, radius, lat, lon):

        # Compute unit-intensity spot at (0, 0)
        z = self._spot_fac * (self._spot_theta - radius)
        b = 1.0 / (1.0 + tt.exp(-z)) - 1.0
        yT = tt.zeros((1, self.Ny))
        yT = tt.set_subtensor(yT[:, self._spot_idx], tt.dot(self._spot_Bp, b))

        # Rotate in latitude then in longitude
        yT = self.dotR(yT, np.array(1.0), np.array(0.0), np.array(0.0), lat)
        yT = self.dotR(yT, np.array(0.0), np.array(1.0), np.array(0.0), -lon)

        # Reshape and we're done
        if self.nw is None:
            y = yT.reshape((-1,)) * contrast
        else:
            y = yT.reshape((-1, 1)) * tt.reshape(contrast, (1, -1))

        return y


class OpsLD(object):
    """Class housing Theano operations for limb-darkened maps."""

    def __init__(self, ydeg, udeg, fdeg, nw, reflected=False, **kwargs):
        # Sanity checks
        assert ydeg == fdeg == 0
        assert reflected is False

        # Ingest kwargs
        self.udeg = udeg
        self.nw = nw

        # Set up the ops
        self._get_cl = GetClOp()
        self._limbdark = LimbDarkOp()
        self._LimbDarkIsPhysical = LDPhysicalOp(_c_ops.nroots)

    @autocompile
    def limbdark_is_physical(self, u):
        """Return True if the limb darkening profile is physical."""
        return self._LimbDarkIsPhysical(u)

    @autocompile
    def intensity(self, mu, u):
        """Compute the intensity at a set of points."""
        if self.udeg == 0:
            mu_f = tt.reshape(mu, (-1,))
            intensity = tt.ones_like(mu_f)
            intensity = tt.set_subtensor(intensity[tt.isnan(mu_f)], np.nan)
            return intensity
        else:
            basis = tt.reshape(1.0 - mu, (-1, 1)) ** np.arange(self.udeg + 1)
            return -tt.dot(basis, u)

    @autocompile
    def flux(self, xo, yo, zo, ro, u):
        """Compute the light curve."""
        # Initialize flat light curve
        flux = tt.ones_like(xo)

        # Compute the occultation mask
        b = tt.sqrt(xo ** 2 + yo ** 2)
        b_occ = tt.invert(tt.ge(b, 1.0 + ro) | tt.le(zo, 0.0) | tt.eq(ro, 0.0))
        i_occ = tt.arange(b.size)[b_occ]

        # Get the Agol `c` coefficients
        c = self._get_cl(u)
        if self.udeg == 0:
            c_norm = c / (np.pi * c[0])
        else:
            c_norm = c / (np.pi * (c[0] + 2 * c[1] / 3))

        # Compute the occultation flux
        los = zo[i_occ]
        r = ro * tt.ones_like(los)
        flux = tt.set_subtensor(
            flux[i_occ], self._limbdark(c_norm, b[i_occ], r, los)[0]
        )
        return flux

    @autocompile
    def X(self, theta, xo, yo, zo, ro, inc, obl, u, f):
        """
        Convenience function for integration of limb-darkened maps
        with the ``System`` class. The design matrix for limb-darkened
        maps is just a column vector equal to the total flux, since the
        spherical harmonic coefficient vector is ``[1.0]``.

        """
        flux = self.flux(xo, yo, zo, ro, u)
        X = tt.reshape(flux, (-1, 1))
        return X

    @autocompile
    def render(self, res, projection, theta, inc, obl, y, u, f):
        """Render the map on a Cartesian grid."""
        nframes = tt.shape(theta)[0]
        image = self.render_ld(res, u)
        return tt.tile(image, (nframes, 1, 1))

    @autocompile
    def render_ld(self, res, u):
        """Simplified version of `render` w/o the extra params.

        The method `render` requires a bunch of dummy params for
        compatibility with the `System` class. This method is a
        convenience method for use in the `Map` class.
        """
        # See NOTE on tt.mgrid bug in `compute_ortho_grid`
        dx = 2.0 / (res - 0.01)
        y, x = tt.mgrid[-1:1:dx, -1:1:dx]

        x = tt.reshape(x, [1, -1])
        y = tt.reshape(y, [1, -1])
        mu = tt.sqrt(1 - x ** 2 - y ** 2)

        # Compute the intensity
        intensity = self.intensity(mu, u)

        # We need the shape to be (nframes, npix, npix)
        return tt.reshape(intensity, (1, res, res))

    @autocompile
    def set_map_vector(self, vector, inds, vals):
        """Set the elements of the theano map coefficient tensor."""
        res = tt.set_subtensor(vector[inds], vals * tt.ones_like(vector[inds]))
        return res


class OpsRV(OpsYlm):
    """Class housing Theano operations for radial velocity maps."""

    @autocompile
    def compute_rv_filter(self, inc, obl, veq, alpha):
        """Compute the radial velocity field Ylm multiplicative filter."""
        # Define some angular quantities
        cosi = tt.cos(inc)
        sini = tt.sin(inc)
        cosl = tt.cos(obl)
        sinl = tt.sin(obl)
        A = sini * cosl
        B = sini * sinl
        C = cosi

        # Compute the Ylm expansion of the RV field
        return (
            tt.reshape(
                [
                    0,
                    veq
                    * np.sqrt(3)
                    * B
                    * (-(A ** 2) * alpha - B ** 2 * alpha - C ** 2 * alpha + 5)
                    / 15,
                    0,
                    veq
                    * np.sqrt(3)
                    * A
                    * (-(A ** 2) * alpha - B ** 2 * alpha - C ** 2 * alpha + 5)
                    / 15,
                    0,
                    0,
                    0,
                    0,
                    0,
                    veq * alpha * np.sqrt(70) * B * (3 * A ** 2 - B ** 2) / 70,
                    veq
                    * alpha
                    * 2
                    * np.sqrt(105)
                    * C
                    * (-(A ** 2) + B ** 2)
                    / 105,
                    veq
                    * alpha
                    * np.sqrt(42)
                    * B
                    * (A ** 2 + B ** 2 - 4 * C ** 2)
                    / 210,
                    0,
                    veq
                    * alpha
                    * np.sqrt(42)
                    * A
                    * (A ** 2 + B ** 2 - 4 * C ** 2)
                    / 210,
                    veq * alpha * 4 * np.sqrt(105) * A * B * C / 105,
                    veq * alpha * np.sqrt(70) * A * (A ** 2 - 3 * B ** 2) / 70,
                ],
                [-1],
            )
            * np.pi
        )

    @autocompile
    def rv(self, theta, xo, yo, zo, ro, inc, obl, y, u, veq, alpha):
        """Compute the observed radial velocity anomaly."""
        # Compute the velocity-weighted intensity
        f = self.compute_rv_filter(inc, obl, veq, alpha)
        Iv = self.flux(theta, xo, yo, zo, ro, inc, obl, y, u, f)

        # Compute the inverse of the intensity
        f0 = tt.zeros_like(f)
        f0 = tt.set_subtensor(f0[0], np.pi)
        I = self.flux(theta, xo, yo, zo, ro, inc, obl, y, u, f0)
        invI = tt.ones((1,)) / I
        invI = tt.where(tt.isinf(invI), 0.0, invI)

        # The RV signal is just the product
        return Iv * invI


class OpsReflected(OpsYlm):
    """Class housing Theano operations for reflected light maps."""

    def __init__(self, *args, **kwargs):
        super(OpsReflected, self).__init__(*args, reflected=True, **kwargs)
        self._rT = rTReflectedOp(self._c_ops.rTReflected, self._c_ops.N)
        self._sT = sTReflectedOp(self._c_ops.sTReflected, self._c_ops.N)
        self._A1Big = ts.as_sparse_variable(self._c_ops.A1Big)

        # Compute grid on unit disk with ~source_npts points
        source_npts = kwargs.get("source_npts", 1)
        if source_npts <= 1:
            self.source_dx = np.array([0.0])
            self.source_dy = np.array([0.0])
            self.source_dz = np.array([0.0])
            self.source_npts = 1
        else:
            N = int(2 + np.sqrt(source_npts * 4 / np.pi))
            dx = np.linspace(-1, 1, N)
            dx, dy = np.meshgrid(dx, dx)
            dz = 1 - dx ** 2 - dy ** 2
            self.source_dx = dx[dz > 0].flatten()
            self.source_dy = dy[dz > 0].flatten()
            self.source_dz = dz[dz > 0].flatten()
            self.source_npts = len(self.source_dx)

        # NOTE: dz is *negative*, since the source is actually
        # *closer* to the body than in the point approximation
        self.source_dx = tt.as_tensor_variable(self.source_dx)
        self.source_dy = tt.as_tensor_variable(self.source_dy)
        self.source_dz = tt.as_tensor_variable(-self.source_dz)

        # Oren-Nayar (1994) intensity profile (for rendering)
        self._OrenNayar = OrenNayarOp(self._c_ops.OrenNayarPolynomial)
        self._pTON94 = pTOp(self._c_ops.pT, _c_ops.STARRY_OREN_NAYAR_DEG)

    @property
    def A1Big(self):
        return self._A1Big

    @autocompile
    def rT(self, b, sigr):
        return self._rT(b, sigr)[0]

    @autocompile
    def sT(self, b, theta, bo, ro, sigr):
        return self._sT(b, theta, bo, ro, sigr)[0]

    @autocompile
    def intensity(
        self,
        lat,
        lon,
        y,
        u,
        f,
        xs,
        ys,
        zs,
        Rs,
        theta,
        ld,
        sigr,
        on94_exact,
        illuminate,
    ):
        """Compute the intensity at a series of lat-lon points on the surface."""
        # Get the Cartesian points
        xpt, ypt, zpt = self.latlon_to_xyz(lat, lon)

        # Compute the polynomial basis at the point
        pT = self.pT(xpt, ypt, zpt)

        # Transform the map to the polynomial basis
        A1y = ts.dot(self.A1, y)

        # Apply the filter
        if self.filter:
            u0 = tt.zeros_like(u)
            u0 = tt.set_subtensor(u0[0], -1.0)
            A1y = ifelse(
                ld, tt.dot(self.F(u, f), A1y), tt.dot(self.F(u0, f), A1y)
            )

        # Dot the polynomial into the basis
        intensity = tt.shape_padright(tt.dot(pT, A1y))

        # Weight the intensity by the illumination
        xyz = tt.concatenate(
            (
                tt.reshape(xpt, [1, -1]),
                tt.reshape(ypt, [1, -1]),
                tt.reshape(zpt, [1, -1]),
            )
        )
        I = self.compute_illumination(xyz, xs, ys, zs, Rs, sigr, on94_exact)

        # Add an extra dimension for the wavelength
        if self.nw is not None:
            I = tt.shape_padaxis(I, 1)

        # Weight the image by the illumination
        # NOTE: The factor of `pi` ensures the correct normalization.
        # This is *different* from the derivation in the paper, but it's
        # due to the fact that the in starry we normalize the spherical
        # harmonics in a slightly strange way (they're normalized so that
        # the integral of Y_{0,0} over the unit sphere is 4, not 4pi).
        # This is useful for thermal light maps, where the flux from a map
        # with Y_{0,0} = 1 is *unity*. But it messes up things for reflected
        # light maps, so we need to account for that here.
        intensity = tt.switch(
            tt.isnan(intensity),
            intensity,
            ifelse(
                illuminate, intensity * I, np.pi * intensity * tt.ones_like(I)
            ),
        )

        return intensity

    @autocompile
    def unweighted_intensity(self, lat, lon, y, u, f, theta, ld):
        """
        Compute the intensity in the absence of an illumination source
        (i.e., the albedo).

        """
        # Get the Cartesian points
        xpt, ypt, zpt = self.latlon_to_xyz(lat, lon)

        # Compute the polynomial basis at the point
        pT = self.pT(xpt, ypt, zpt)

        # Transform the map to the polynomial basis
        A1y = ts.dot(self.A1, y)

        # Apply the filter
        if self.filter:
            u0 = tt.zeros_like(u)
            u0 = tt.set_subtensor(u0[0], -1.0)
            A1y = ifelse(
                ld, tt.dot(self.F(u, f), A1y), tt.dot(self.F(u0, f), A1y)
            )

        # Dot the polynomial into the basis.
        # NOTE: The factor of `pi` ensures the correct normalization.
        # This is *different* from the derivation in the paper, but it's
        # due to the fact that the in starry we normalize the spherical
        # harmonics in a slightly strange way (they're normalized so that
        # the integral of Y_{0,0} over the unit sphere is 4, not 4pi).
        # This is useful for thermal light maps, where the flux from a map
        # with Y_{0,0} = 1 is *unity*. But it messes up things for reflected
        # light maps, so we need to account for that here.
        return np.pi * tt.dot(pT, A1y)

    @autocompile
    def X_point_source(
        self, theta, xs, ys, zs, xo, yo, zo, ro, inc, obl, u, f, sigr
    ):
        """Compute the light curve design matrix for a point source."""
        # Determine shapes
        rows = theta.shape[0]
        cols = (self.ydeg + 1) ** 2
        X = tt.zeros((rows, cols))

        # Compute the occultation mask
        bo = tt.sqrt(xo ** 2 + yo ** 2)
        b_rot = tt.ge(bo, 1.0 + ro) | tt.le(zo, 0.0) | tt.eq(ro, 0.0)
        b_occ = tt.invert(b_rot)
        i_rot = tt.arange(bo.size)[b_rot]
        i_occ = tt.arange(bo.size)[b_occ]

        # Compute filter operator
        if self.filter:
            F = self.F(u, f)

        # Terminator
        r2 = xs ** 2 + ys ** 2 + zs ** 2
        b_term = -zs / tt.sqrt(r2)
        theta_term = tt.arctan2(xo, yo) - tt.arctan2(xs, ys)

        # Rotation operator
        rT = self.rT(b_term[i_rot], sigr)
        if self.filter:
            rTA1 = ts.dot(tt.dot(rT, F), self.A1)
        else:
            rTA1 = ts.dot(rT, self.A1)
        theta_z = tt.arctan2(xs[i_rot], ys[i_rot])
        rTA1Rz = self.tensordotRz(rTA1, theta_z)
        X = tt.set_subtensor(
            X[i_rot], self.right_project(rTA1Rz, inc, obl, theta[i_rot])
        )

        # Occultation + rotation operator
        sT = self.sT(b_term[i_occ], theta_term[i_occ], bo[i_occ], ro, sigr)
        sTA = ts.dot(sT, self.A)
        theta_z = tt.arctan2(xo[i_occ], yo[i_occ])
        sTAR = self.tensordotRz(sTA, theta_z)
        if self.filter:
            A1InvFA1 = ts.dot(ts.dot(self.A1Inv, F), self.A1)
            sTAR = tt.dot(sTAR, A1InvFA1)
        X = tt.set_subtensor(
            X[i_occ], self.right_project(sTAR, inc, obl, theta[i_occ])
        )

        # Weight by the distance to the source.
        # NOTE: In the paper, we divide by `pi` at this step. But the way
        # we coded things up, starry maps are already implicitly pi-normalized
        # (via the change-of-basis matrix A1), so we don't need to do that here.
        X /= tt.shape_padright(r2)

        # We're done
        return X

    @autocompile
    def X(self, theta, xs, ys, zs, Rs, xo, yo, zo, ro, inc, obl, u, f, sigr):
        """Compute the light curve design matrix."""

        if self.source_npts == 1:

            return self.X_point_source(
                theta, xs, ys, zs, xo, yo, zo, ro, inc, obl, u, f, sigr
            )

        else:

            # The effective size of the star as seen by the planet
            # is smaller. Only include points
            # that fall on this smaller disk.
            rs = tt.sqrt(xs ** 2 + ys ** 2 + zs ** 2)
            Reff = Rs * tt.sqrt(1 - ((Rs - 1) / rs) ** 2)
            dx = tt.shape_padright(Reff) * self.source_dx
            dy = tt.shape_padright(Reff) * self.source_dy
            # Note that the star is *closer* to the planet, hence the - sign
            dz = -tt.sqrt(Rs ** 2 - dx ** 2 - dy ** 2)

            # Compute the illumination for each point on the source disk
            X = self.X_point_source(
                tt.reshape(
                    tt.shape_padright(theta) + tt.zeros_like(dx), (-1,)
                ),
                tt.reshape(tt.shape_padright(xs) + dx, (-1,)),
                tt.reshape(tt.shape_padright(ys) + dy, (-1,)),
                tt.reshape(tt.shape_padright(zs) + dz, (-1,)),
                tt.reshape(tt.shape_padright(xo) + tt.zeros_like(dx), (-1,)),
                tt.reshape(tt.shape_padright(yo) + tt.zeros_like(dx), (-1,)),
                tt.reshape(tt.shape_padright(zo) + tt.zeros_like(dx), (-1,)),
                ro,
                inc,
                obl,
                u,
                f,
                sigr,
            )
            X = tt.reshape(X, (X.shape[0], -1))

            # Point source approximation
            X0 = self.X_point_source(
                theta, xs, ys, zs, xo, yo, zo, ro, inc, obl, u, f, sigr
            )
            X0 = tt.reshape(X0, (X0.shape[0], -1))

            # Average over each profile if Rs != 0
            return ifelse(
                Rs > 0,
                ifelse(
                    tt.shape(theta)[0] > 0,
                    (
                        tt.sum(
                            tt.reshape(
                                X, (tt.shape(theta)[0], self.source_npts, -1)
                            ),
                            axis=1,
                        )
                        / self.source_npts
                    ),
                    tt.zeros_like(X),
                ),
                X0,
            )

    @autocompile
    def flux(
        self, theta, xs, ys, zs, Rs, xo, yo, zo, ro, inc, obl, y, u, f, sigr
    ):
        """Compute the reflected light curve."""
        return tt.dot(
            self.X(
                theta, xs, ys, zs, Rs, xo, yo, zo, ro, inc, obl, u, f, sigr
            ),
            y,
        )

    @autocompile
    def flux_point_source(
        self, theta, xs, ys, zs, xo, yo, zo, ro, inc, obl, y, u, f, sigr
    ):
        """Compute the reflected light curve for a point source."""
        return tt.dot(
            self.X_point_source(
                theta, xs, ys, zs, xo, yo, zo, ro, inc, obl, u, f, sigr
            ),
            y,
        )

    @autocompile
    def render(
        self,
        res,
        projection,
        illuminate,
        theta,
        inc,
        obl,
        y,
        u,
        f,
        xs,
        ys,
        zs,
        Rs,
        sigr,
        on94_exact,
    ):
        """Render the map on a Cartesian grid."""
        # Compute the Cartesian grid
        xyz = ifelse(
            tt.eq(projection, STARRY_RECTANGULAR_PROJECTION),
            self.compute_rect_grid(res)[-1],
            ifelse(
                tt.eq(projection, STARRY_MOLLWEIDE_PROJECTION),
                self.compute_moll_grid(res)[-1],
                self.compute_ortho_grid(res)[-1],
            ),
        )

        # If orthographic, rotate the map to the correct frame
        if self.nw is None:
            Ry = ifelse(
                tt.eq(projection, STARRY_ORTHOGRAPHIC_PROJECTION),
                self.left_project(
                    tt.transpose(tt.tile(y, [theta.shape[0], 1])),
                    inc,
                    obl,
                    theta,
                ),
                tt.transpose(tt.tile(y, [theta.shape[0], 1])),
            )
        else:
            Ry = ifelse(
                tt.eq(projection, STARRY_ORTHOGRAPHIC_PROJECTION),
                self.left_project(y, inc, obl, tt.tile(theta[0], self.nw)),
                y,
            )

        # Transform to polynomials
        A1Ry = ts.dot(self.A1, Ry)

        # Apply the filter *only if orthographic*
        f0 = tt.zeros_like(f)
        f0 = tt.set_subtensor(f0[0], np.pi)
        u0 = tt.zeros_like(u)
        u0 = tt.set_subtensor(u0[0], -1.0)
        A1Ry = ifelse(
            tt.eq(projection, STARRY_ORTHOGRAPHIC_PROJECTION),
            tt.dot(self.F(u, f), A1Ry),
            tt.dot(self.F(u0, f0), A1Ry),
        )

        # Dot the polynomial into the basis
        pT = self.pT(xyz[0], xyz[1], xyz[2])
        image = tt.dot(pT, A1Ry)

        # Compute the illumination profile
        I = self.compute_illumination(xyz, xs, ys, zs, Rs, sigr, on94_exact)

        # Add an extra dimension for the wavelength
        if self.nw is not None:
            I = tt.repeat(I, self.nw, 1)

        # Weight the image by the illumination
        # NOTE: The factor of `pi` ensures the correct normalization.
        # This is *different* from the derivation in the paper, but it's
        # due to the fact that the in starry we normalize the spherical
        # harmonics in a slightly strange way (they're normalized so that
        # the integral of Y_{0,0} over the unit sphere is 4, not 4pi).
        # This is useful for thermal light maps, where the flux from a map
        # with Y_{0,0} = 1 is *unity*. But it messes up things for reflected
        # light maps, so we need to account for that here.
        image = ifelse(
            illuminate,
            tt.switch(tt.isnan(image), image, image * I),
            np.pi * image,
        )

        # We need the shape to be (nframes, npix, npix)
        return tt.reshape(image, [res, res, -1]).dimshuffle(2, 0, 1)

    @autocompile
    def compute_illumination_point_source(
        self, xyz, xs, ys, zs, sigr, on94_exact
    ):
        """Compute the illumination profile for a point source."""
        # Get cos(theta_i)
        x = tt.shape_padright(xyz[0])
        y = tt.shape_padright(xyz[1])
        z = tt.shape_padright(xyz[2])
        r2 = xs ** 2 + ys ** 2 + zs ** 2
        b = -zs / tt.sqrt(r2)  # semi-minor axis of terminator
        invsr = 1.0 / tt.sqrt(xs ** 2 + ys ** 2)
        cosw = ys * invsr
        sinw = -xs * invsr
        xrot = x * cosw + y * sinw
        yrot = -x * sinw + y * cosw
        bc = tt.sqrt(1.0 - b ** 2)
        cos_thetai = bc * yrot - b * z

        # Check for special cases
        cos_thetai = tt.switch(
            tt.eq(tt.abs_(b), 1.0),
            tt.switch(
                tt.eq(b, 1.0), tt.zeros_like(cos_thetai), z  # midnight  # noon
            ),
            cos_thetai,
        )
        # Set night to zero
        cos_thetai = tt.switch(
            tt.gt(cos_thetai, 0.0), cos_thetai, tt.zeros_like(cos_thetai)
        )

        # Lambertian intensity
        I_lamb = cos_thetai

        # Polynomial approximation to the Oren-Nayar (1994)
        # intensity with the nightside masked
        pT = self._pTON94(xyz[0], xyz[1], xyz[2])
        theta = -tt.arctan2(xs, ys)
        p = self._OrenNayar(b, theta, sigr)
        I_on94_approx = tt.dot(pT, p)
        I_on94_approx = tt.switch(
            tt.gt(cos_thetai, 0.0), I_on94_approx, tt.zeros_like(I_on94_approx)
        )

        # "Exact" Oren-Nayar (1994) intensity from their Equation (30)
        f1 = -b / z - cos_thetai
        f2 = -b / cos_thetai - z
        f = cos_thetai * tt.maximum(0, tt.minimum(f1, f2))
        sig2 = sigr ** 2
        A = 1.0 - 0.5 * sig2 / (sig2 + 0.33)
        B = 0.45 * sig2 / (sig2 + 0.09)
        I_on94_exact = A * cos_thetai + B * f

        # Select the function we want
        I = ifelse(
            sigr > 0, ifelse(on94_exact, I_on94_exact, I_on94_approx), I_lamb
        )

        # Weight by the distance to the source.
        # NOTE: In the paper, we divide by `pi` at this step. But the way
        # we coded things up, starry maps are already implicitly pi-normalized
        # (via the change-of-basis matrix A1), so we don't need to do that here.
        I /= tt.shape_padleft(r2)
        return I

    @autocompile
    def compute_illumination(self, xyz, xs, ys, zs, Rs, sigr, on94_exact):
        """Compute the illumination profile when rendering maps."""

        if self.source_npts == 1:

            return self.compute_illumination_point_source(
                xyz, xs, ys, zs, sigr, on94_exact
            )

        else:

            # The effective size of the star as seen by the planet
            # is smaller. Only include points
            # that fall on this smaller disk.
            rs = tt.sqrt(xs ** 2 + ys ** 2 + zs ** 2)
            Reff = Rs * tt.sqrt(1 - ((Rs - 1) / rs) ** 2)
            dx = tt.shape_padright(Reff) * self.source_dx
            dy = tt.shape_padright(Reff) * self.source_dy
            # Note that the star is *closer* to the planet, hence the - sign
            dz = -tt.sqrt(Rs ** 2 - dx ** 2 - dy ** 2)

            # Compute the illumination for each point on the source disk
            I = self.compute_illumination_point_source(
                xyz,
                tt.reshape(tt.shape_padright(xs) + dx, (-1,)),
                tt.reshape(tt.shape_padright(ys) + dy, (-1,)),
                tt.reshape(tt.shape_padright(zs) + dz, (-1,)),
                sigr,
                on94_exact,
            )
            I = tt.reshape(I, (-1, tt.shape(xs)[0], self.source_npts))

            # Average over each profile
            return tt.sum(I, axis=2) / self.source_npts


class OpsSystem(object):
    """Class housing ops for modeling Keplerian systems."""

    def __init__(
        self,
        primary,
        secondaries,
        reflected=False,
        rv=False,
        light_delay=False,
        texp=None,
        oversample=7,
        order=0,
    ):
        # System members
        self.primary = primary
        self.secondaries = secondaries
        self._reflected = reflected
        self._rv = rv
        self.nw = self.primary._map.nw
        self.light_delay = light_delay
        self.texp = texp
        self.oversample = oversample
        self.order = order

    @autocompile
    def position(
        self,
        t,
        pri_m,
        pri_t0,
        sec_m,
        sec_t0,
        sec_porb,
        sec_ecc,
        sec_w,
        sec_Omega,
        sec_iorb,
    ):
        """Compute the Cartesian positions of all bodies."""
        orbit = exoplanet.orbits.KeplerianOrbit(
            period=sec_porb,
            t0=sec_t0,
            incl=sec_iorb,
            ecc=sec_ecc,
            omega=sec_w,
            Omega=sec_Omega,
            m_planet=sec_m,
            m_star=pri_m,
            r_star=1.0,  # doesn't matter
        )
        # Position of the primary
        if len(self.secondaries) > 1:
            x_pri, y_pri, z_pri = tt.sum(orbit.get_star_position(t), axis=-1)
        else:
            x_pri, y_pri, z_pri = orbit.get_star_position(t)

        # Positions of the secondaries
        try:
            x_sec, y_sec, z_sec = orbit.get_planet_position(
                t, light_delay=self.light_delay
            )
        except TypeError:
            if self.light_delay:
                logger.warn(
                    "This version of `exoplanet` does not model light delays."
                )
            x_sec, y_sec, z_sec = orbit.get_planet_position(t)

        # Concatenate them
        x = tt.transpose(tt.concatenate((x_pri, x_sec), axis=-1))
        y = tt.transpose(tt.concatenate((y_pri, y_sec), axis=-1))
        z = tt.transpose(tt.concatenate((z_pri, z_sec), axis=-1))

        return x, y, z

    @autocompile
    def X(
        self,
        t,
        pri_r,
        pri_m,
        pri_prot,
        pri_t0,
        pri_theta0,
        pri_amp,
        pri_inc,
        pri_obl,
        pri_u,
        pri_f,
        sec_r,
        sec_m,
        sec_prot,
        sec_t0,
        sec_theta0,
        sec_porb,
        sec_ecc,
        sec_w,
        sec_Omega,
        sec_iorb,
        sec_amp,
        sec_inc,
        sec_obl,
        sec_u,
        sec_f,
        sec_sigr,
    ):
        """Compute the system light curve design matrix."""
        # Exposure time integration?
        if self.texp != 0.0:

            texp = tt.as_tensor_variable(self.texp)
            oversample = int(self.oversample)
            oversample += 1 - oversample % 2
            stencil = np.ones(oversample)

            # Construct the exposure time integration stencil
            if self.order == 0:
                dt = np.linspace(-0.5, 0.5, 2 * oversample + 1)[1:-1:2]
            elif self.order == 1:
                dt = np.linspace(-0.5, 0.5, oversample)
                stencil[1:-1] = 2
            elif self.order == 2:
                dt = np.linspace(-0.5, 0.5, oversample)
                stencil[1:-1:2] = 4
                stencil[2:-1:2] = 2
            else:
                raise ValueError("Parameter `order` must be <= 2")
            stencil /= np.sum(stencil)

            if texp.ndim == 0:
                dt = texp * dt
            else:
                dt = tt.shape_padright(texp) * dt
            t = tt.shape_padright(t) + dt
            t = tt.reshape(t, (-1,))

        # Compute the relative positions of all bodies
        orbit = exoplanet.orbits.KeplerianOrbit(
            period=sec_porb,
            t0=sec_t0,
            incl=sec_iorb,
            ecc=sec_ecc,
            omega=sec_w,
            Omega=sec_Omega,
            m_planet=sec_m,
            m_star=pri_m,
            r_star=pri_r,
        )
        try:
            x, y, z = orbit.get_relative_position(
                t, light_delay=self.light_delay
            )
        except TypeError:
            if self.light_delay:
                logger.warn(
                    "This version of `exoplanet` does not model light delays."
                )
            x, y, z = orbit.get_relative_position(t)

        # Get all rotational phases
        pri_prot = ifelse(
            tt.eq(pri_prot, 0.0), math.to_tensor(np.inf), pri_prot
        )
        theta_pri = (2 * np.pi) / pri_prot * (t - pri_t0) + pri_theta0
        sec_prot = tt.switch(
            tt.eq(sec_prot, 0.0), math.to_tensor(np.inf), sec_prot
        )
        theta_sec = (2 * np.pi) / tt.shape_padright(sec_prot) * (
            tt.shape_padleft(t) - tt.shape_padright(sec_t0)
        ) + tt.shape_padright(sec_theta0)

        # Compute all the phase curves
        phase_pri = pri_amp * self.primary.map.ops.X(
            theta_pri,
            tt.zeros_like(t),
            tt.zeros_like(t),
            tt.zeros_like(t),
            math.to_tensor(0.0),
            pri_inc,
            pri_obl,
            pri_u,
            pri_f,
        )
        if self._reflected:
            phase_sec = [
                pri_amp
                * sec_amp[i]
                * sec.map.ops.X(
                    theta_sec[i],
                    -x[:, i] / sec_r[i],
                    -y[:, i] / sec_r[i],
                    -z[:, i] / sec_r[i],
                    pri_r / sec_r[i],  # scaled source radius
                    tt.zeros_like(x[:, i]),
                    tt.zeros_like(x[:, i]),
                    tt.zeros_like(x[:, i]),
                    math.to_tensor(0.0),  # occultor of zero radius
                    sec_inc[i],
                    sec_obl[i],
                    sec_u[i],
                    sec_f[i],
                    sec_sigr[i],
                )
                for i, sec in enumerate(self.secondaries)
            ]
        else:
            phase_sec = [
                sec_amp[i]
                * sec.map.ops.X(
                    theta_sec[i],
                    -x[:, i],
                    -y[:, i],
                    -z[:, i],
                    math.to_tensor(0.0),  # occultor of zero radius
                    sec_inc[i],
                    sec_obl[i],
                    sec_u[i],
                    sec_f[i],
                )
                for i, sec in enumerate(self.secondaries)
            ]

        # Compute any occultations
        occ_pri = tt.zeros_like(phase_pri)
        occ_sec = [tt.zeros_like(ps) for ps in phase_sec]

        # Compute the period if we were given a semi-major axis
        sec_porb = tt.switch(
            tt.eq(sec_porb, 0.0),
            (G_grav * (pri_m + sec_m) * sec_porb ** 2 / (4 * np.pi ** 2))
            ** (1.0 / 3),
            sec_porb,
        )

        # Compute transits across the primary
        for i, _ in enumerate(self.secondaries):
            xo = x[:, i] / pri_r
            yo = y[:, i] / pri_r
            zo = z[:, i] / pri_r
            ro = sec_r[i] / pri_r
            b = tt.sqrt(xo ** 2 + yo ** 2)
            b_occ = tt.invert(
                tt.ge(b, 1.0 + ro) | tt.le(zo, 0.0) | tt.eq(ro, 0.0)
            )
            idx = tt.arange(b.shape[0])[b_occ]
            occ_pri = tt.set_subtensor(
                occ_pri[idx],
                occ_pri[idx]
                + pri_amp
                * self.primary.map.ops.X(
                    theta_pri[idx],
                    xo[idx],
                    yo[idx],
                    zo[idx],
                    ro,
                    pri_inc,
                    pri_obl,
                    pri_u,
                    pri_f,
                )
                - phase_pri[idx],
            )

        # Compute occultations by the primary
        for i, sec in enumerate(self.secondaries):

            xo = -x[:, i] / sec_r[i]
            yo = -y[:, i] / sec_r[i]
            zo = -z[:, i] / sec_r[i]
            ro = pri_r / sec_r[i]
            b = tt.sqrt(xo ** 2 + yo ** 2)
            b_occ = tt.invert(
                tt.ge(b, 1.0 + ro) | tt.le(zo, 0.0) | tt.eq(ro, 0.0)
            )
            idx = tt.arange(b.shape[0])[b_occ]
            if self._reflected:
                occ_sec[i] = tt.set_subtensor(
                    occ_sec[i][idx],
                    occ_sec[i][idx]
                    + pri_amp
                    * sec_amp[i]
                    * sec.map.ops.X(
                        theta_sec[i, idx],
                        xo[idx],  # the primary is both the source...
                        yo[idx],
                        zo[idx],
                        ro,
                        xo[idx],  # ... and the occultor
                        yo[idx],
                        zo[idx],
                        ro,
                        sec_inc[i],
                        sec_obl[i],
                        sec_u[i],
                        sec_f[i],
                        sec_sigr[i],
                    )
                    - phase_sec[i][idx],
                )
            else:
                occ_sec[i] = tt.set_subtensor(
                    occ_sec[i][idx],
                    occ_sec[i][idx]
                    + sec_amp[i]
                    * sec.map.ops.X(
                        theta_sec[i, idx],
                        xo[idx],
                        yo[idx],
                        zo[idx],
                        ro,
                        sec_inc[i],
                        sec_obl[i],
                        sec_u[i],
                        sec_f[i],
                    )
                    - phase_sec[i][idx],
                )

        # Compute secondary-secondary occultations
        for i, sec in enumerate(self.secondaries):
            for j, _ in enumerate(self.secondaries):
                if i == j:
                    continue
                xo = (-x[:, i] + x[:, j]) / sec_r[i]
                yo = (-y[:, i] + y[:, j]) / sec_r[i]
                zo = (-z[:, i] + z[:, j]) / sec_r[i]
                ro = sec_r[j] / sec_r[i]
                b = tt.sqrt(xo ** 2 + yo ** 2)
                b_occ = tt.invert(
                    tt.ge(b, 1.0 + ro) | tt.le(zo, 0.0) | tt.eq(ro, 0.0)
                )
                idx = tt.arange(b.shape[0])[b_occ]
                if self._reflected:
                    xs = -x[:, i] / sec_r[i]
                    ys = -y[:, i] / sec_r[i]
                    zs = -z[:, i] / sec_r[i]
                    occ_sec[i] = tt.set_subtensor(
                        occ_sec[i][idx],
                        occ_sec[i][idx]
                        + sec_amp[i]
                        * pri_amp
                        * sec.map.ops.X(
                            theta_sec[i, idx],
                            xs[idx],  # the primary is the source
                            ys[idx],
                            zs[idx],
                            pri_r / sec_r[i],
                            xo[idx],  # another secondary is the occultor
                            yo[idx],
                            zo[idx],
                            ro,
                            sec_inc[i],
                            sec_obl[i],
                            sec_u[i],
                            sec_f[i],
                            sec_sigr[i],
                        )
                        - phase_sec[i][idx],
                    )
                else:
                    occ_sec[i] = tt.set_subtensor(
                        occ_sec[i][idx],
                        occ_sec[i][idx]
                        + sec_amp[i]
                        * sec.map.ops.X(
                            theta_sec[i, idx],
                            xo[idx],
                            yo[idx],
                            zo[idx],
                            ro,
                            sec_inc[i],
                            sec_obl[i],
                            sec_u[i],
                            sec_f[i],
                        )
                        - phase_sec[i][idx],
                    )

        # Concatenate the design matrices
        X_pri = phase_pri + occ_pri
        X_sec = [ps + os for ps, os in zip(phase_sec, occ_sec)]
        X = tt.horizontal_stack(X_pri, *X_sec)

        # Sum and return
        if self.texp == 0.0:

            return X

        else:

            stencil = tt.shape_padright(tt.shape_padleft(stencil, 1), 1)
            return tt.sum(
                stencil * tt.reshape(X, (-1, self.oversample, X.shape[1])),
                axis=1,
            )

    @autocompile
    def rv(
        self,
        t,
        pri_r,
        pri_m,
        pri_prot,
        pri_t0,
        pri_theta0,
        pri_amp,
        pri_inc,
        pri_obl,
        pri_y,
        pri_u,
        pri_alpha,
        pri_veq,
        sec_r,
        sec_m,
        sec_prot,
        sec_t0,
        sec_theta0,
        sec_porb,
        sec_ecc,
        sec_w,
        sec_Omega,
        sec_iorb,
        sec_amp,
        sec_inc,
        sec_obl,
        sec_y,
        sec_u,
        sec_alpha,
        sec_sigr,
        sec_veq,
        keplerian,
    ):
        """Compute the observed system radial velocity (RV maps only)."""
        # TODO: This method is currently very inefficient, as it
        # calls `X` twice per call and instantiates an `orbit`
        # instance up to three separate times per call. We should
        # re-code the logic from `X()` in here to optimize it.

        # Compute the RV filter
        pri_f = self.primary.map.ops.compute_rv_filter(
            pri_inc, pri_obl, pri_veq, pri_alpha
        )
        sec_f = tt.as_tensor_variable(
            [
                sec.map.ops.compute_rv_filter(
                    sec_inc[k], sec_obl[k], sec_veq[k], sec_alpha[k]
                )
                for k, sec in enumerate(self.secondaries)
            ]
        )

        # Compute the identity filter
        pri_f0 = tt.zeros_like(pri_f)
        pri_f0 = tt.set_subtensor(pri_f0[0], np.pi)
        sec_f0 = tt.as_tensor_variable([pri_f0 for sec in self.secondaries])

        # Compute the two design matrices
        X = self.X(
            t,
            pri_r,
            pri_m,
            pri_prot,
            pri_t0,
            pri_theta0,
            pri_amp,
            pri_inc,
            pri_obl,
            pri_u,
            pri_f,
            sec_r,
            sec_m,
            sec_prot,
            sec_t0,
            sec_theta0,
            sec_porb,
            sec_ecc,
            sec_w,
            sec_Omega,
            sec_iorb,
            sec_amp,
            sec_inc,
            sec_obl,
            sec_u,
            sec_f,
            sec_sigr,
        )

        X0 = self.X(
            t,
            pri_r,
            pri_m,
            pri_prot,
            pri_t0,
            pri_theta0,
            pri_amp,
            pri_inc,
            pri_obl,
            pri_u,
            pri_f0,
            sec_r,
            sec_m,
            sec_prot,
            sec_t0,
            sec_theta0,
            sec_porb,
            sec_ecc,
            sec_w,
            sec_Omega,
            sec_iorb,
            sec_amp,
            sec_inc,
            sec_obl,
            sec_u,
            sec_f0,
            sec_sigr,
        )

        # Get the indices of X corresponding to each body
        pri_inds = np.arange(0, self.primary.map.Ny, dtype=int)
        sec_inds = [None for sec in self.secondaries]
        n = self.primary.map.Ny
        for i, sec in enumerate(self.secondaries):
            sec_inds[i] = np.arange(n, n + sec.map.Ny, dtype=int)
            n += sec.map.Ny

        # Compute the integral of the velocity-weighted intensity
        Iv = tt.as_tensor_variable(
            [tt.dot(X[:, pri_inds], pri_y)]
            + [
                tt.dot(X[:, sec_inds[n]], sec_y[n])
                for n in range(len(self.secondaries))
            ]
        )

        # Compute the inverse of the integral of the intensity
        invI = tt.as_tensor_variable(
            [tt.ones((1,)) / tt.dot(X0[:, pri_inds], pri_y)]
            + [
                tt.ones((1,)) / tt.dot(X0[:, sec_inds[n]], sec_y[n])
                for n in range(len(self.secondaries))
            ]
        )
        invI = tt.where(tt.isinf(invI), 0.0, invI)

        # The RV anomaly is just the product
        rv = Iv * invI

        # Compute the Keplerian RV
        orbit = exoplanet.orbits.KeplerianOrbit(
            period=sec_porb,
            t0=sec_t0,
            incl=sec_iorb,
            ecc=sec_ecc,
            omega=sec_w,
            Omega=sec_Omega,
            m_planet=sec_m,
            m_star=pri_m,
            r_star=pri_r,
        )
        return ifelse(
            keplerian,
            tt.inc_subtensor(
                rv[1:],
                tt.transpose(
                    orbit.get_radial_velocity(
                        t, output_units=units.m / units.s
                    )
                ),
            ),
            rv,
        )

    @autocompile
    def render(
        self,
        t,
        res,
        pri_r,
        pri_m,
        pri_prot,
        pri_t0,
        pri_theta0,
        pri_inc,
        pri_obl,
        pri_y,
        pri_u,
        pri_f,
        sec_r,
        sec_m,
        sec_prot,
        sec_t0,
        sec_theta0,
        sec_porb,
        sec_ecc,
        sec_w,
        sec_Omega,
        sec_iorb,
        sec_inc,
        sec_obl,
        sec_y,
        sec_u,
        sec_f,
        sec_sigr,
    ):
        """Render all of the bodies in the system."""
        # Compute the relative positions of all bodies
        orbit = exoplanet.orbits.KeplerianOrbit(
            period=sec_porb,
            t0=sec_t0,
            incl=sec_iorb,
            ecc=sec_ecc,
            omega=sec_w,
            Omega=sec_Omega,
            m_planet=sec_m,
            m_star=pri_m,
            r_star=pri_r,
        )
        try:
            x, y, z = orbit.get_relative_position(
                t, light_delay=self.light_delay
            )
        except TypeError:
            if self.light_delay:
                logger.warn(
                    "This version of `exoplanet` does not model light delays."
                )
            x, y, z = orbit.get_relative_position(t)

        # Get all rotational phases
        pri_prot = ifelse(
            tt.eq(pri_prot, 0.0), math.to_tensor(np.inf), pri_prot
        )
        theta_pri = (2 * np.pi) / pri_prot * (t - pri_t0) + pri_theta0
        sec_prot = tt.switch(
            tt.eq(sec_prot, 0.0), math.to_tensor(np.inf), sec_prot
        )
        theta_sec = (2 * np.pi) / tt.shape_padright(sec_prot) * (
            tt.shape_padleft(t) - tt.shape_padright(sec_t0)
        ) + tt.shape_padright(sec_theta0)

        # Compute all the maps
        img_pri = self.primary.map.ops.render(
            res,
            STARRY_ORTHOGRAPHIC_PROJECTION,
            theta_pri,
            pri_inc,
            pri_obl,
            pri_y,
            pri_u,
            pri_f,
        )
        if self._reflected:
            img_sec = tt.as_tensor_variable(
                [
                    sec.map.ops.render(
                        res,
                        STARRY_ORTHOGRAPHIC_PROJECTION,
                        1,
                        theta_sec[i],
                        sec_inc[i],
                        sec_obl[i],
                        sec_y[i],
                        sec_u[i],
                        sec_f[i],
                        -x[:, i],
                        -y[:, i],
                        -z[:, i],
                        pri_r / sec_r[i],
                        sec_sigr[i],
                        0,  # use approx Oren-Nayar intensity
                    )
                    for i, sec in enumerate(self.secondaries)
                ]
            )
        else:
            img_sec = tt.as_tensor_variable(
                [
                    sec.map.ops.render(
                        res,
                        STARRY_ORTHOGRAPHIC_PROJECTION,
                        theta_sec[i],
                        sec_inc[i],
                        sec_obl[i],
                        sec_y[i],
                        sec_u[i],
                        sec_f[i],
                    )
                    for i, sec in enumerate(self.secondaries)
                ]
            )

        # Return the images and secondary orbital positions
        return img_pri, img_sec, x, y, z
