# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import minimize
import theano
from theano import gof
import theano.tensor as tt

__all__ = ["minimizeOp"]


class minimizeOp(tt.Op):
    """Find the global minimum of the map intensity.

    Returns the tuple `(lat, lon, I)`.
    """

    def __init__(self, intensity, P, ydeg, udeg, fdeg):
        self.intensity = intensity
        self.P = P
        self.ydeg = ydeg
        self.udeg = udeg
        self.fdeg = fdeg
        self._do_setup = True
        self.fac = 1
        self.result = None

    def setup(self, fac=1):
        # Don't setup unless the user actually calls this function,
        # since there's quite a bit of overhead
        if self._do_setup or (fac != self.fac):

            # The grid seach resolution
            self.fac = fac
            res = int(self.fac * self.ydeg)

            # Create the grid
            lon_grid = np.linspace(-np.pi, np.pi, res)
            lat_grid = np.arccos(1 - np.arange(2 * res + 1) / res) - np.pi / 2
            lon_grid, lat_grid = np.meshgrid(lon_grid, lat_grid)
            lon_grid, lat_grid = lon_grid.flatten(), lat_grid.flatten()

            # Add some noise to try to counteract aliasing
            s = np.random.RandomState(0)
            lon_grid += (0.1 / res) * s.randn(len(lon_grid))
            lat_grid += (0.1 / res) * s.randn(len(lat_grid))

            # Evaluate the intensities
            self.P_grid = self.P(
                tt.as_tensor_variable(lat_grid),
                tt.as_tensor_variable(lon_grid),
            ).eval()
            self.lat_grid = lat_grid
            self.lon_grid = lon_grid

            # Set up the cost & grad function for the nonlinear solver
            u0 = np.zeros(self.udeg + 1)
            u0[0] = -1.0
            u0 = tt.as_tensor_variable(u0)
            f0 = np.zeros((self.fdeg + 1) ** 2)
            f0[0] = np.pi
            f0 = tt.as_tensor_variable(f0)
            latlon = tt.dvector()
            y = tt.dvector()
            self.I = theano.function(
                [latlon, y],
                [
                    self.intensity(latlon[0], latlon[1], y, u0, f0)[0],
                    *theano.grad(
                        self.intensity(latlon[0], latlon[1], y, u0, f0)[0],
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
        ind = np.argmin(I_grid)
        x0 = np.array([self.lat_grid[ind], self.lon_grid[ind]])

        # Minimize
        result = minimize(self.I, x0, args=(y,), jac=True)
        outputs[0][0] = result.x[0]
        outputs[1][0] = result.x[1]
        outputs[2][0] = result.fun

        # Save
        self.result = result
