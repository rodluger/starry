# -*- coding: utf-8 -*-
from .. import config
from .. import _c_ops
from .integration import sTOp, rTReflectedOp
from .rotation import dotROp, tensordotRzOp
from .filter import FOp
from .misc import spotYlmOp, pTOp
from .utils import *
import theano
import theano.tensor as tt
import theano.sparse as ts
from theano.ifelse import ifelse
import numpy as np
import logging
from astropy import units, constants

try:
    # starry requires exoplanet >= v0.2.0
    from packaging import version
    import exoplanet

    if version.parse(exoplanet.__version__) < version.parse("0.2.0"):
        exoplanet = None
except ModuleNotFoundError:
    exoplanet = None


# Gravitational constant in internal units
G_grav = constants.G.to(units.R_sun ** 3 / units.M_sun / units.day ** 2).value


__all__ = ["Ops", "OpsReflected", "OpsRV", "OpsSystem", "OpsRVSystem"]


class Ops(object):
    """
    Everything in radians here.
    Everything is a Theano operation.

    """

    def __init__(self, ydeg, udeg, fdeg, nw, quiet=False, **kwargs):
        """

        """
        # Logging
        if quiet:
            logger.setLevel(logging.WARNING)
        else:
            logger.setLevel(logging.INFO)

        # Instantiate the C++ Ops
        self.ydeg = ydeg
        self.udeg = udeg
        self.fdeg = fdeg
        self.deg = ydeg + udeg + fdeg
        self.filter = (fdeg > 0) or (udeg > 0)
        self._c_ops = _c_ops.Ops(ydeg, udeg, fdeg)
        self.nw = nw
        if config.lazy:
            self.cast = to_tensor
        else:
            self.cast = to_array

        # Solution vectors
        self.sT = sTOp(self._c_ops.sT, self._c_ops.N)
        self.rT = tt.shape_padleft(tt.as_tensor_variable(self._c_ops.rT))
        self.rTA1 = tt.shape_padleft(tt.as_tensor_variable(self._c_ops.rTA1))

        # Change of basis matrices
        self.A = ts.as_sparse_variable(self._c_ops.A)
        self.A1 = ts.as_sparse_variable(self._c_ops.A1)
        self.A1Inv = ts.as_sparse_variable(self._c_ops.A1Inv)

        # Rotation operations
        self.tensordotRz = tensordotRzOp(self._c_ops.tensordotRz)
        self.dotR = dotROp(self._c_ops.dotR)

        # Filter
        # TODO: This should be sparse!!!
        self.F = FOp(self._c_ops.F, self._c_ops.N, self._c_ops.Ny)

        # Misc
        self.spotYlm = spotYlmOp(self._c_ops.spotYlm, self.ydeg, self.nw)
        self.pT = pTOp(self._c_ops.pT, self.deg)

    @autocompile(
        "X",
        tt.dvector(),
        tt.dvector(),
        tt.dvector(),
        tt.dvector(),
        tt.dscalar(),
        tt.dscalar(),
        tt.dscalar(),
        tt.dvector(),
        tt.dvector(),
    )
    def X(self, theta, xo, yo, zo, ro, inc, obl, u, f):
        """

        """
        # Compute the occultation mask
        b = tt.sqrt(xo ** 2 + yo ** 2)
        b_rot = tt.ge(b, 1.0 + ro) | tt.le(zo, 0.0) | tt.eq(ro, 0.0)
        b_occ = tt.invert(b_rot)
        i_rot = tt.arange(b.size)[b_rot]
        i_occ = tt.arange(b.size)[b_occ]

        # Determine shapes
        rows = theta.shape[0]
        cols = self.rTA1.shape[1]

        # Compute filter operator
        if self.filter:
            F = self.F(u, f)

        # Rotation operator
        if self.filter:
            rTA1 = ts.dot(tt.dot(self.rT, F), self.A1)
        else:
            rTA1 = self.rTA1
        X_rot = tt.set_subtensor(
            tt.zeros((rows, cols))[i_rot],
            self.right_project(rTA1, inc, obl, theta[i_rot]),
        )

        # Occultation + rotation operator
        sT = self.sT(b[i_occ], ro)
        sTA = ts.dot(sT, self.A)
        theta_z = tt.arctan2(xo[i_occ], yo[i_occ])
        sTAR = self.tensordotRz(sTA, theta_z)
        if self.filter:
            A1InvFA1 = ts.dot(ts.dot(self.A1Inv, F), self.A1)
            sTAR = tt.dot(sTAR, A1InvFA1)
        X_occ = tt.set_subtensor(
            tt.zeros((rows, cols))[i_occ],
            self.right_project(sTAR, inc, obl, theta[i_occ]),
        )

        return X_rot + X_occ

    @autocompile(
        "flux",
        tt.dvector(),
        tt.dvector(),
        tt.dvector(),
        tt.dvector(),
        tt.dscalar(),
        tt.dscalar(),
        tt.dscalar(),
        DynamicType("tt.dvector() if instance.nw is None else tt.dmatrix()"),
        tt.dvector(),
        tt.dvector(),
    )
    def flux(self, theta, xo, yo, zo, ro, inc, obl, y, u, f):
        """

        """
        return tt.dot(
            self.X(theta, xo, yo, zo, ro, inc, obl, u, f, no_compile=True), y
        )

    @autocompile("P", tt.dvector(), tt.dvector(), tt.dvector())
    def P(self, lat, lon):
        """
        Pixelization matrix, no filters or illumination.

        """
        # Get the Cartesian points
        xpt, ypt, zpt = self.latlon_to_xyz(lat, lon)

        # Compute the polynomial basis at the point
        pT = self.pT(xpt, ypt, zpt)[:, : (self.ydeg + 1) ** 2]

        # Transform to the Ylm basis
        pTA1 = ts.dot(pT, self.A1)

        # We're done
        return pTA1

    @autocompile(
        "intensity",
        tt.dvector(),
        tt.dvector(),
        DynamicType("tt.dvector() if instance.nw is None else tt.dmatrix()"),
        tt.dvector(),
        tt.dvector(),
    )
    def intensity(self, lat, lon, y, u, f):
        """

        """
        # Get the Cartesian points
        xpt, ypt, zpt = self.latlon_to_xyz(lat, lon)

        # Compute the polynomial basis at the point
        pT = self.pT(xpt, ypt, zpt)

        # Transform the map to the polynomial basis
        A1y = ts.dot(self.A1, y)

        # Apply the filter
        if self.filter:
            A1y = tt.dot(self.F(u, f), A1y)

        # Dot the polynomial into the basis
        return tt.dot(pT, A1y)

    @autocompile(
        "render",
        tt.iscalar(),
        tt.iscalar(),
        tt.dvector(),
        tt.dscalar(),
        tt.dscalar(),
        DynamicType("tt.dvector() if instance.nw is None else tt.dmatrix()"),
        tt.dvector(),
        tt.dvector(),
    )
    def render(self, res, projection, theta, inc, obl, y, u, f):
        """

        """
        # Compute the Cartesian grid
        xyz = ifelse(
            tt.eq(projection, STARRY_RECTANGULAR_PROJECTION),
            self.compute_rect_grid(res),
            self.compute_ortho_grid(res),
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

    @autocompile(
        "add_spot",
        DynamicType("tt.dvector() if instance.nw is None else tt.dmatrix()"),
        DynamicType("tt.dscalar() if instance.nw is None else tt.dvector()"),
        tt.dvector(),
        tt.dscalar(),
        tt.dscalar(),
        tt.dscalar(),
    )
    def add_spot(self, y, L, amp, sigma, lat, lon):
        """

        """
        y_new = y + self.spotYlm(amp, sigma, lat, lon)
        L_new = L * y_new[0]
        y_new /= y_new[0]
        return y_new, L_new

    def compute_ortho_grid(self, res):
        """
        Compute the polynomial basis on the plane of the sky.

        """
        dx = 2.0 / res
        y, x = tt.mgrid[-1:1:dx, -1:1:dx]
        x = tt.reshape(x, [1, -1])
        y = tt.reshape(y, [1, -1])
        z = tt.sqrt(1 - x ** 2 - y ** 2)
        return tt.concatenate((x, y, z))

    def compute_rect_grid(self, res):
        """
        Compute the polynomial basis on a rectangular lat/lon grid.

        """
        dx = np.pi / res
        lat, lon = tt.mgrid[
            -np.pi / 2 : np.pi / 2 : dx, -3 * np.pi / 2 : np.pi / 2 : 2 * dx
        ]
        x = tt.reshape(tt.cos(lat) * tt.cos(lon), [1, -1])
        y = tt.reshape(tt.cos(lat) * tt.sin(lon), [1, -1])
        z = tt.reshape(tt.sin(lat), [1, -1])
        R = RAxisAngle([1, 0, 0], -np.pi / 2)
        return tt.dot(R, tt.concatenate((x, y, z)))

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
                    tt.as_tensor_variable(0.0),
                    -(0.5 * np.pi - inc),
                ),
                tt.as_tensor_variable(0.0),
                tt.as_tensor_variable(0.0),
                tt.as_tensor_variable(1.0),
                obl,
            ),
            tt.as_tensor_variable(1.0),
            tt.as_tensor_variable(0.0),
            tt.as_tensor_variable(0.0),
            -0.5 * np.pi,
        )

        # Rotate to the correct phase
        M = self.tensordotRz(M, theta)

        # Rotate to the polar frame
        M = self.dotR(
            M,
            tt.as_tensor_variable(1.0),
            tt.as_tensor_variable(0.0),
            tt.as_tensor_variable(0.0),
            0.5 * np.pi,
        )

        return M

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
            tt.as_tensor_variable(1.0),
            tt.as_tensor_variable(0.0),
            tt.as_tensor_variable(0.0),
            -0.5 * np.pi,
        )

        # Rotate to the correct phase
        MT = self.tensordotRz(MT, -theta)

        # Rotate to the sky frame
        # TODO: Do this in a single compound rotation
        MT = self.dotR(
            self.dotR(
                self.dotR(
                    MT,
                    tt.as_tensor_variable(1.0),
                    tt.as_tensor_variable(0.0),
                    tt.as_tensor_variable(0.0),
                    0.5 * np.pi,
                ),
                tt.as_tensor_variable(0.0),
                tt.as_tensor_variable(0.0),
                tt.as_tensor_variable(1.0),
                -obl,
            ),
            -tt.cos(obl),
            -tt.sin(obl),
            tt.as_tensor_variable(0.0),
            (0.5 * np.pi - inc),
        )

        return tt.transpose(MT)

    def set_map_vector(self, vector, inds, vals):
        """

        """
        res = tt.set_subtensor(vector[inds], vals * tt.ones_like(vector[inds]))
        return res

    def latlon_to_xyz(self, lat, lon):
        """

        """
        R1 = VectorRAxisAngle([1.0, 0.0, 0.0], -lat)
        R2 = VectorRAxisAngle([0.0, 1.0, 0.0], lon)
        R = tt.batched_dot(R2, R1)
        xyz = tt.transpose(tt.dot(R, [0.0, 0.0, 1.0]))
        return xyz[0], xyz[1], xyz[2]


class OpsRV(Ops):
    """

    """

    @autocompile(
        "compute_rv_filter",
        tt.dscalar(),
        tt.dscalar(),
        tt.dscalar(),
        tt.dscalar(),
    )
    def compute_rv_filter(self, inc, obl, veq, alpha):
        """

        """
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
                    * (-A ** 2 * alpha - B ** 2 * alpha - C ** 2 * alpha + 5)
                    / 15,
                    0,
                    veq
                    * np.sqrt(3)
                    * A
                    * (-A ** 2 * alpha - B ** 2 * alpha - C ** 2 * alpha + 5)
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
                    * (-A ** 2 + B ** 2)
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

    @autocompile(
        "rv",
        tt.dvector(),
        tt.dvector(),
        tt.dvector(),
        tt.dvector(),
        tt.dscalar(),
        tt.dscalar(),
        tt.dscalar(),
        DynamicType("tt.dvector() if instance.nw is None else tt.dmatrix()"),
        tt.dvector(),
        tt.dscalar(),
        tt.dscalar(),
    )
    def rv(self, theta, xo, yo, zo, ro, inc, obl, y, u, veq, alpha):
        """

        """
        # Compute the velocity-weighted intensity
        f = self.compute_rv_filter(inc, obl, veq, alpha, no_compile=True)
        Iv = self.flux(
            theta, xo, yo, zo, ro, inc, obl, y, u, f, no_compile=True
        )

        # Compute the inverse of the intensity
        f0 = tt.zeros_like(f)
        f0 = tt.set_subtensor(f0[0], np.pi)
        I = self.flux(
            theta, xo, yo, zo, ro, inc, obl, y, u, f0, no_compile=True
        )
        invI = tt.ones((1,)) / I
        invI = tt.where(tt.isinf(invI), 0.0, invI)

        # The RV signal is just the product
        return Iv * invI


class OpsReflected(Ops):
    """

    """

    def __init__(self, *args, **kwargs):
        """

        """
        super(OpsReflected, self).__init__(*args, **kwargs)
        self.rT = rTReflectedOp(self._c_ops.rTReflected, self._c_ops.N)
        self.A1Big = ts.as_sparse_variable(self._c_ops.A1Big)

    def compute_illumination(self, xyz, source):
        """

        """
        b = -source[:, 2]
        invsr = 1.0 / tt.sqrt(source[:, 0] ** 2 + source[:, 1] ** 2)
        cosw = source[:, 1] * invsr
        sinw = -source[:, 0] * invsr
        xrot = (
            tt.shape_padright(xyz[0]) * cosw + tt.shape_padright(xyz[1]) * sinw
        )
        yrot = (
            -tt.shape_padright(xyz[0]) * sinw
            + tt.shape_padright(xyz[1]) * cosw
        )
        I = tt.sqrt(1.0 - b ** 2) * yrot - b * tt.shape_padright(xyz[2])
        I = tt.switch(
            tt.eq(tt.abs_(b), 1.0),
            tt.switch(
                tt.eq(b, 1.0),
                tt.zeros_like(I),  # midnight
                tt.shape_padright(xyz[2]),  # noon
            ),
            I,
        )
        I = tt.switch(tt.gt(I, 0.0), I, tt.zeros_like(I))
        return I

    @autocompile(
        "intensity",
        tt.dvector(),
        tt.dvector(),
        DynamicType("tt.dvector() if instance.nw is None else tt.dmatrix()"),
        tt.dvector(),
        tt.dvector(),
        tt.dmatrix(),
    )
    def intensity(self, lat, lon, y, u, f, source):
        """

        """
        # Get the Cartesian points
        xpt, ypt, zpt = self.latlon_to_xyz(lat, lon)

        # Compute the polynomial basis at the point
        pT = self.pT(xpt, ypt, zpt)

        # Transform the map to the polynomial basis
        A1y = ts.dot(self.A1, y)

        # Apply the filter
        A1y = tt.dot(self.F(u, f), A1y)

        # Dot the polynomial into the basis
        intensity = tt.dot(pT, A1y)

        # Weight the intensity by the illumination
        xyz = tt.concatenate(
            (
                tt.reshape(xpt, [1, -1]),
                tt.reshape(ypt, [1, -1]),
                tt.reshape(xpt, [1, -1]),
            )
        )
        I = self.compute_illumination(xyz, source)
        intensity = tt.switch(tt.isnan(intensity), intensity, intensity * I)
        return intensity

    @autocompile(
        "X",
        tt.dvector(),
        tt.dvector(),
        tt.dvector(),
        tt.dvector(),
        tt.dscalar(),
        tt.dscalar(),
        tt.dscalar(),
        tt.dvector(),
        tt.dvector(),
        tt.dmatrix(),
    )
    def X(self, theta, xo, yo, zo, ro, inc, obl, u, f, source):
        """

        """
        # Determine shapes
        rows = theta.shape[0]
        cols = (self.ydeg + 1) ** 2

        # Compute the occultation mask
        b = tt.sqrt(xo ** 2 + yo ** 2)
        b_rot = tt.ge(b, 1.0 + ro) | tt.le(zo, 0.0) | tt.eq(ro, 0.0)
        b_occ = tt.invert(b_rot)
        i_occ = tt.arange(b.size)[b_occ]

        # Compute the semi-minor axis of the terminator
        # and the reflectance integrals
        source /= tt.reshape(source.norm(2, axis=1), [-1, 1])
        bterm = -source[:, 2]
        rT = self.rT(bterm)

        # Transform to Ylms and rotate on the sky plane
        rTA1 = ts.dot(rT, self.A1Big)
        theta_z = tt.arctan2(source[:, 0], source[:, 1])
        rTA1Rz = self.tensordotRz(rTA1, theta_z)

        # Apply limb darkening?
        F = self.F(u, f)
        A1InvFA1 = ts.dot(ts.dot(self.A1Inv, F), self.A1)
        rTA1Rz = tt.dot(rTA1Rz, A1InvFA1)

        # Rotate to the correct phase
        X = self.right_project(rTA1Rz, inc, obl, theta)

        # TODO: Implement occultations in reflected light
        # Throw error if there's an occultation
        X = X + RaiseValuerErrorIfOp(
            "Occultations in reflected light not yet implemented."
        )(b_occ.any())

        # We're done
        return X

    @autocompile(
        "flux",
        tt.dvector(),
        tt.dvector(),
        tt.dvector(),
        tt.dvector(),
        tt.dscalar(),
        tt.dscalar(),
        tt.dscalar(),
        DynamicType("tt.dvector() if instance.nw is None else tt.dmatrix()"),
        tt.dvector(),
        tt.dvector(),
        tt.dmatrix(),
    )
    def flux(self, theta, xo, yo, zo, ro, inc, obl, y, u, f, source):
        """

        """
        return tt.dot(
            self.X(
                theta, xo, yo, zo, ro, inc, obl, u, f, source, no_compile=True
            ),
            y,
        )

    @autocompile(
        "render",
        tt.iscalar(),
        tt.iscalar(),
        tt.dvector(),
        tt.dscalar(),
        tt.dscalar(),
        DynamicType("tt.dvector() if instance.nw is None else tt.dmatrix()"),
        tt.dvector(),
        tt.dvector(),
        tt.dmatrix(),
    )
    def render(self, res, projection, theta, inc, obl, y, u, f, source):
        """
        
        """
        # Compute the Cartesian grid
        xyz = ifelse(
            tt.eq(projection, STARRY_RECTANGULAR_PROJECTION),
            self.compute_rect_grid(res),
            self.compute_ortho_grid(res),
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
        image = tt.dot(pT, A1Ry)

        # Compute the illumination profile
        source /= tt.reshape(source.norm(2, axis=1), [-1, 1])
        I = self.compute_illumination(xyz, source)

        # Weight the image by the illumination
        image = tt.switch(tt.isnan(image), image, image * I)

        # We need the shape to be (nframes, npix, npix)
        return tt.reshape(image, [res, res, -1]).dimshuffle(2, 0, 1)


class OpsSystem(object):
    """

    """

    def __init__(self, primary, secondaries, reflected=False, quiet=False):
        """

        """
        # Logging
        if quiet:
            logger.setLevel(logging.WARNING)
        else:
            logger.setLevel(logging.INFO)

        # System members
        self.primary = primary
        self.secondaries = secondaries
        self.reflected = reflected
        self.nw = self.primary._map.nw

        # Require exoplanet
        assert exoplanet is not None, "This class requires exoplanet >= 0.2.0."

    @autocompile(
        "position",
        tt.dvector(),  # t
        # -- primary --
        tt.dscalar(),  # m
        tt.dscalar(),  # t0
        # -- secondaries --
        tt.dvector(),  # m
        tt.dvector(),  # t0
        tt.dvector(),  # porb
        tt.dvector(),  # a
        tt.dvector(),  # ecc
        tt.dvector(),  # w
        tt.dvector(),  # Omega
        tt.dvector(),  # iorb
    )
    def position(
        self,
        t,
        pri_m,
        pri_t0,
        sec_m,
        sec_t0,
        sec_porb,
        sec_a,
        sec_ecc,
        sec_w,
        sec_Omega,
        sec_iorb,
    ):
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
            m_planet_units=units.Msun,
        )
        # Position of the primary
        if len(self.secondaries) > 1:
            x_pri, y_pri, z_pri = tt.sum(orbit.get_star_position(t), axis=-1)
        else:
            x_pri, y_pri, z_pri = orbit.get_star_position(t)

        # Positions of the secondaries
        x_sec, y_sec, z_sec = orbit.get_planet_position(t)

        # Concatenate them
        x = tt.transpose(tt.concatenate((x_pri, x_sec), axis=-1))
        y = tt.transpose(tt.concatenate((y_pri, y_sec), axis=-1))
        z = tt.transpose(tt.concatenate((z_pri, z_sec), axis=-1))

        return x, y, z

    @autocompile(
        "X",
        tt.dvector(),  # t
        # -- primary --
        tt.dscalar(),  # r
        tt.dscalar(),  # m
        tt.dscalar(),  # prot
        tt.dscalar(),  # t0
        tt.dscalar(),  # theta0
        DynamicType(
            "tt.dscalar() if instance.nw is None else tt.dvector()"
        ),  # L
        tt.dscalar(),  # inc
        tt.dscalar(),  # obl
        tt.dvector(),  # u
        tt.dvector(),  # f
        # -- secondaries --
        tt.dvector(),  # r
        tt.dvector(),  # m
        tt.dvector(),  # prot
        tt.dvector(),  # t0
        tt.dvector(),  # theta0
        tt.dvector(),  # porb
        tt.dvector(),  # a
        tt.dvector(),  # ecc
        tt.dvector(),  # w
        tt.dvector(),  # Omega
        tt.dvector(),  # iorb
        DynamicType(
            "tt.dvector() if instance.nw is None else tt.dmatrix()"
        ),  # L
        tt.dvector(),  # inc
        tt.dvector(),  # obl
        tt.dmatrix(),  # u
        tt.dmatrix(),  # f
    )
    def X(
        self,
        t,
        pri_r,
        pri_m,
        pri_prot,
        pri_t0,
        pri_theta0,
        pri_L,
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
        sec_a,
        sec_ecc,
        sec_w,
        sec_Omega,
        sec_iorb,
        sec_L,
        sec_inc,
        sec_obl,
        sec_u,
        sec_f,
    ):
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
            m_planet_units=units.Msun,
        )
        x, y, z = orbit.get_relative_position(t)

        # Compute the position of the illumination source (the primary)
        # if we're doing things in reflected light
        if self.reflected:
            source = [
                [
                    tt.transpose(
                        tt.as_tensor_variable([-x[:, i], -y[:, i], -z[:, i]])
                    )
                ]
                for i, sec in enumerate(self.secondaries)
            ]
        else:
            source = [[] for sec in self.secondaries]

        # Get all rotational phases
        theta_pri = (2 * np.pi) / pri_prot * (t - pri_t0) - pri_theta0
        theta_sec = (2 * np.pi) / tt.shape_padright(sec_prot) * (
            tt.shape_padleft(t) - tt.shape_padright(sec_t0)
        ) - tt.shape_padright(sec_theta0)

        # Compute all the phase curves
        phase_pri = pri_L * self.primary.map.ops.X(
            theta_pri,
            tt.zeros_like(t),
            tt.zeros_like(t),
            tt.zeros_like(t),
            to_tensor(0.0),
            pri_inc,
            pri_obl,
            pri_u,
            pri_f,
            no_compile=True,
        )
        phase_sec = tt.as_tensor_variable(
            [
                sec_L[i]
                * sec.map.ops.X(
                    theta_sec[i],
                    tt.zeros_like(t),
                    tt.zeros_like(t),
                    tt.zeros_like(t),
                    to_tensor(0.0),
                    sec_inc[i],
                    sec_obl[i],
                    sec_u[i],
                    sec_f[i],
                    *source[i],
                    no_compile=True,
                )
                for i, sec in enumerate(self.secondaries)
            ]
        )

        # Compute any occultations
        occ_pri = tt.zeros_like(phase_pri)
        occ_sec = tt.zeros_like(phase_sec)

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
                + pri_L
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
                    no_compile=True,
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
            if self.reflected:
                source_occ = [
                    [
                        source[0][i][b_occ]
                        for i, _ in enumerate(self.secondaries)
                    ]
                ]
            else:
                source_occ = source
            idx = tt.arange(b.shape[0])[b_occ]
            occ_sec = tt.set_subtensor(
                occ_sec[i, idx],
                occ_sec[i, idx]
                + sec_L[i]
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
                    *source_occ[i],
                    no_compile=True,
                )
                - phase_sec[i, idx],
            )

        # Compute secondary-secondary occultations
        for i, sec in enumerate(self.secondaries):
            for j, _ in enumerate(self.secondaries):
                if i == j:
                    continue
                xo = (x[:, i] - x[:, j]) / sec_r[i]
                yo = (y[:, i] - y[:, j]) / sec_r[i]
                zo = (z[:, i] - z[:, j]) / sec_r[i]
                ro = sec_r[j] / sec_r[i]
                b = tt.sqrt(xo ** 2 + yo ** 2)
                b_occ = tt.invert(
                    tt.ge(b, 1.0 + ro) | tt.le(zo, 0.0) | tt.eq(ro, 0.0)
                )
                if self.reflected:
                    source_occ = [
                        [
                            source[0][i][b_occ]
                            for i, _ in enumerate(self.secondaries)
                        ]
                    ]
                else:
                    source_occ = source
                idx = tt.arange(b.shape[0])[b_occ]
                occ_sec = tt.set_subtensor(
                    occ_sec[i, idx],
                    occ_sec[i, idx]
                    + sec_L[i]
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
                        *source_occ[i],
                        no_compile=True,
                    )
                    - phase_sec[i, idx],
                )

        # Concatenate the design matrices & return
        X_pri = phase_pri + occ_pri
        X_sec = phase_sec + occ_sec
        X_sec = tt.reshape(tt.swapaxes(X_sec, 0, 1), (X_sec.shape[1], -1))
        return tt.horizontal_stack(X_pri, X_sec)

    @autocompile(
        "flux",
        tt.dvector(),  # t
        # -- primary --
        tt.dscalar(),  # r
        tt.dscalar(),  # m
        tt.dscalar(),  # prot
        tt.dscalar(),  # t0
        tt.dscalar(),  # theta0
        DynamicType(
            "tt.dscalar() if instance.nw is None else tt.dvector()"
        ),  # L
        tt.dscalar(),  # inc
        tt.dscalar(),  # obl
        DynamicType(
            "tt.dvector() if instance.nw is None else tt.dmatrix()"
        ),  # y
        tt.dvector(),  # u
        tt.dvector(),  # f
        # -- secondaries --
        tt.dvector(),  # r
        tt.dvector(),  # m
        tt.dvector(),  # prot
        tt.dvector(),  # t0
        tt.dvector(),  # theta0
        tt.dvector(),  # porb
        tt.dvector(),  # a
        tt.dvector(),  # ecc
        tt.dvector(),  # w
        tt.dvector(),  # Omega
        tt.dvector(),  # iorb
        DynamicType(
            "tt.dvector() if instance.nw is None else tt.dmatrix()"
        ),  # L
        tt.dvector(),  # inc
        tt.dvector(),  # obl
        DynamicType(
            "tt.dmatrix() if instance.nw is None else tt.dtensor3()"
        ),  # y
        tt.dmatrix(),  # u
        tt.dmatrix(),  # f
    )
    def flux(
        self,
        t,
        pri_r,
        pri_m,
        pri_prot,
        pri_t0,
        pri_theta0,
        pri_L,
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
        sec_a,
        sec_ecc,
        sec_w,
        sec_Omega,
        sec_iorb,
        sec_L,
        sec_inc,
        sec_obl,
        sec_y,
        sec_u,
        sec_f,
    ):
        X = self.X(
            t,
            pri_r,
            pri_m,
            pri_prot,
            pri_t0,
            pri_theta0,
            pri_L,
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
            sec_a,
            sec_ecc,
            sec_w,
            sec_Omega,
            sec_iorb,
            sec_L,
            sec_inc,
            sec_obl,
            sec_u,
            sec_f,
            no_compile=True,
        )
        y = tt.concatenate((pri_y, tt.reshape(sec_y, (-1,))))
        return tt.dot(X, y)


class OpsRVSystem(OpsSystem):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Radial velocity mode not yet implemented.")
