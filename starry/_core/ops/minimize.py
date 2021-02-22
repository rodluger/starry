# -*- coding: utf-8 -*-
from ...compat import Apply
import numpy as np
from scipy.optimize import minimize
import theano
import theano.tensor as tt
from theano.configparser import change_flags


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
        self.bounds = None

    def setup(self, oversample=1, ntries=1, bounds=None):
        self.ntries = ntries

        # Don't setup unless the user actually calls this function,
        # since there's quite a bit of overhead
        if self._do_setup or (oversample != self.oversample):

            self.oversample = oversample

            # Create the lat-lon grid
            # TODO: Use a mollweide grid instead of
            # random samples!
            # Require at least `oversample * l ** 2 points`
            s = np.random.RandomState(0)
            npts = oversample * self.ydeg ** 2
            self.lat_grid = np.arccos(2 * s.rand(npts) - 1) - np.pi / 2
            self.lon_grid = (s.rand(npts) - 0.5) * 2 * np.pi

            # Restrict grid in latitude/longitude to a certain range
            if bounds is not None:
                self.bounds = (
                    (bounds[0][0] * np.pi / 180, bounds[0][1] * np.pi / 180),
                    (bounds[1][0] * np.pi / 180, bounds[1][1] * np.pi / 180),
                )

                mask_lat = np.logical_and(
                    self.lat_grid > self.bounds[0][0],
                    self.lat_grid < self.bounds[0][1],
                )
                mask_lon = np.logical_and(
                    self.lon_grid > self.bounds[1][0],
                    self.lon_grid < self.bounds[1][1],
                )
                mask_com = np.logical_and(mask_lat, mask_lon)

                self.lat_grid = self.lat_grid[mask_com]
                self.lon_grid = self.lon_grid[mask_com]

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
        return Apply(self, inputs, outputs)

    def infer_shape(self, *args):
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
            result = minimize(
                self.I, x0, args=(y,), jac=True, bounds=self.bounds
            )

            if result.fun < outputs[2][0]:
                outputs[0][0] = np.array(result.x[0])
                outputs[1][0] = np.array(result.x[1])
                outputs[2][0] = np.array(result.fun)

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
        return Apply(self, inputs, outputs)

    def infer_shape(self, *args):
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
