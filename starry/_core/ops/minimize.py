# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import minimize
import theano
from theano import gof
import theano.tensor as tt
from theano.configparser import change_flags

try:
    import healpy as hp
except ImportError:
    hp = None


__all__ = ["minimizeOp", "LDPhysicalOp"]


class minimizeOp(tt.Op):
    """Find the global minimum of the map intensity.

    Returns the tuple `(lat, lon, I)`.

    .. note::
        This op is not very optimized. The idea here is to
        do a coarse grid search, find the minimum, then run
        a quick gradient descent to refine the result.
    """

    def __init__(self, intensity, P, ydeg, udeg, fdeg):
        self.intensity = intensity
        self.P = P
        self.ydeg = ydeg
        self.udeg = udeg
        self.fdeg = fdeg
        self._do_setup = True
        self.oversample = 1
        self.ntries = 1
        self.result = None

    def setup(self, oversample=1, ntries=1):
        self.ntries = ntries

        # Don't setup unless the user actually calls this function,
        # since there's quite a bit of overhead
        if self._do_setup or (oversample != self.oversample):

            self.oversample = oversample

            # Create the grid using healpy if available
            # Require at least `oversample * l ** 2 points`
            s = np.random.RandomState(0)
            if hp is None:
                npts = oversample * self.ydeg ** 2
                self.lat_grid = (
                    np.arccos(2 * s.rand(npts) - 1) * 180.0 / np.pi - 90.0
                )
                self.lon_grid = (s.rand(npts) - 0.5) * 360
            else:
                nside = 1
                while hp.nside2npix(nside) < oversample * self.ydeg ** 2:
                    nside += 1
                theta, phi = hp.pix2ang(
                    nside=nside, ipix=range(hp.nside2npix(nside))
                )
                self.lat_grid = 0.5 * np.pi - theta
                self.lon_grid = phi - np.pi
                # Add a little noise for stability
                self.lat_grid += 1e-4 * s.randn(len(self.lat_grid))
                self.lon_grid += 1e-4 * s.randn(len(self.lon_grid))
            self.P_grid = self.P(
                tt.as_tensor_variable(self.lat_grid),
                tt.as_tensor_variable(self.lon_grid),
            ).eval()

            # Set up the cost & grad function for the nonlinear solver
            u0 = np.zeros(self.udeg + 1)
            u0[0] = -1.0
            u0 = tt.as_tensor_variable(u0)
            f0 = np.zeros((self.fdeg + 1) ** 2)
            f0[0] = np.pi
            f0 = tt.as_tensor_variable(f0)
            latlon = tt.dvector()
            y = tt.dvector()
            with change_flags(compute_test_value="off"):
                self.I = theano.function(
                    [latlon, y],
                    [
                        self.intensity(
                            latlon[0], latlon[1], y, u0, f0, 0.0, 0.0
                        )[0],
                        *theano.grad(
                            self.intensity(
                                latlon[0], latlon[1], y, u0, f0, 0.0, 0.0
                            )[0],
                            [latlon],
                        ),
                    ],
                )
            self._do_setup = False

    def make_node(self, *inputs):
        inputs = [tt.as_tensor_variable(i) for i in inputs]
        outputs = [
            tt.TensorType(inputs[0].dtype, ())(),
            tt.TensorType(inputs[0].dtype, ())(),
            tt.TensorType(inputs[0].dtype, ())(),
        ]
        return gof.Apply(self, inputs, outputs)

    def infer_shape(self, node, shapes):
        return [(), (), ()]

    def perform(self, node, inputs, outputs):

        assert self._do_setup is False, "Must run `setup()` first."

        y = inputs[0]

        # Initial guess
        I_grid = np.dot(self.P_grid, y)
        outputs[2][0] = np.inf

        for n in range(self.ntries):

            ind = np.argmin(I_grid)
            x0 = np.array([self.lat_grid[ind], self.lon_grid[ind]])

            # Minimize
            result = minimize(self.I, x0, args=(y,), jac=True)

            if result.fun < outputs[2][0]:
                outputs[0][0] = result.x[0]
                outputs[1][0] = result.x[1]
                outputs[2][0] = result.fun

            # Prepare for next iteration
            I_grid[ind] = np.inf

        # Save
        self.result = result


class LDPhysicalOp(tt.Op):
    """
    Check whether a limb darkening profile is physical using Sturm's theorem.

    """

    def __init__(self, nroots):
        self.nroots = nroots

    def make_node(self, *inputs):
        inputs = [tt.as_tensor_variable(i) for i in inputs]
        outputs = [tt.bscalar()]
        return gof.Apply(self, inputs, outputs)

    def infer_shape(self, node, shapes):
        return [()]

    def perform(self, node, inputs, outputs):
        u = inputs[0]
        outputs[0][0] = 1

        # Ensure the function is *decreasing* toward the limb
        if u.sum() < -1:
            outputs[0][0] = 0
            return

        # Sturm's theorem on the intensity to ensure positivity
        p = u[::-1]
        if self.nroots(p, 0, 1) > 0:
            outputs[0][0] = 0
            return

        # Sturm's theorem on the derivative to ensure monotonicity
        p = (u[1:] * np.arange(1, len(u)))[::-1]
        if self.nroots(p, 0, 1) > 0:
            outputs[0][0] = 0
            return
