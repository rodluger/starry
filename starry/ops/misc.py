# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np
from scipy.optimize import minimize
import theano
from theano import gof
import theano.tensor as tt
import theano.sparse as ts

__all__ = ["minimizeOp", "spotYlmOp", "pTOp"]


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

    def setup(self):
        # Don't setup unless the user actually calls this function,
        # since there's quite a bit of overhead
        if self._do_setup:
            # Coarse map rendering transform on an equal area lat-lon grid.
            # The maximum number of extrema of a band-limited function
            # on the sphere is l^2 - l + 2 (Kuznetsov & Kholshevnikov 1992)
            # The minimum resolution of the grid must therefore be...
            res = int(
                np.ceil(
                    0.25
                    * (np.sqrt(1 + 8 * (self.ydeg ** 2 - self.ydeg + 2)) - 1)
                )
            )
            lon_grid = np.linspace(-np.pi, np.pi, res)
            lat_grid = np.arccos(1 - np.arange(2 * res + 1) / res) - np.pi / 2
            lon_grid, lat_grid = np.meshgrid(lon_grid, lat_grid)
            lon_grid, lat_grid = lon_grid.flatten(), lat_grid.flatten()
            self.P_grid = self.P(lat_grid, lon_grid, no_compile=True).eval()
            self.lat_grid = lat_grid
            self.lon_grid = lon_grid

            # Set up the cost & grad function for the nonlinear solver
            u0 = np.zeros(self.udeg + 1)
            u0[0] = -1.0
            self.u0 = tt.as_tensor_variable(u0)
            f0 = np.zeros((self.fdeg + 1) ** 2)
            f0[0] = np.pi
            self.f0 = tt.as_tensor_variable(f0)
            latlon = tt.dvector()
            y = tt.dvector()
            self.I = theano.function(
                [latlon, y],
                [
                    self.intensity(
                        latlon[0], latlon[1], y, u0, f0, no_compile=True
                    )[0],
                    *theano.grad(
                        self.intensity(
                            latlon[0], latlon[1], y, u0, f0, no_compile=True
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
        ind = np.argmin(I_grid)
        x0 = np.array([self.lat_grid[ind], self.lon_grid[ind]])

        # Minimize
        result = minimize(self.I, x0, args=(y,), jac=True)
        outputs[0][0] = result.x[0]
        outputs[1][0] = result.x[1]
        outputs[2][0] = result.fun


class spotYlmOp(tt.Op):
    def __init__(self, func, ydeg, nw):
        self.func = func
        self.Ny = (ydeg + 1) ** 2
        self.nw = nw

    def make_node(self, *inputs):
        inputs = [tt.as_tensor_variable(i) for i in inputs]
        if self.nw is None:
            outputs = [tt.TensorType(inputs[0].dtype, (False,))()]
        else:
            outputs = [tt.TensorType(inputs[0].dtype, (False, False))()]
        return gof.Apply(self, inputs, outputs)

    def infer_shape(self, node, shapes):
        if self.nw is None:
            return [(self.Ny,)]
        else:
            return [(self.Ny, self.nw)]

    def perform(self, node, inputs, outputs):
        outputs[0][0] = self.func(*inputs)
        if self.nw is None:
            outputs[0][0] = np.reshape(outputs[0][0], -1)


class pTOp(tt.Op):
    def __init__(self, func, deg):
        self.func = func
        self.deg = deg
        self.N = (deg + 1) ** 2
        self._grad_op = pTGradientOp(self)

    def make_node(self, *inputs):
        inputs = [tt.as_tensor_variable(i) for i in inputs]
        outputs = [tt.TensorType(inputs[0].dtype, (False, False))()]
        return gof.Apply(self, inputs, outputs)

    def infer_shape(self, node, shapes):
        return [[shapes[0][0], self.N]]

    def perform(self, node, inputs, outputs):
        outputs[0][0] = self.func(*inputs)

    def grad(self, inputs, gradients):
        return self._grad_op(*(inputs + gradients))


class pTGradientOp(tt.Op):
    def __init__(self, base_op):
        self.base_op = base_op

        # Pre-compute the gradient factors for x, y, and z
        n = 0
        self.xf = np.zeros(self.base_op.N, dtype=int)
        self.yf = np.zeros(self.base_op.N, dtype=int)
        self.zf = np.zeros(self.base_op.N, dtype=int)
        for l in range(self.base_op.deg + 1):
            for m in range(-l, l + 1):
                mu = l - m
                nu = l + m
                if nu % 2 == 0:
                    if mu > 0:
                        self.xf[n] = mu // 2
                    if nu > 0:
                        self.yf[n] = nu // 2
                else:
                    if mu > 1:
                        self.xf[n] = (mu - 1) // 2
                    if nu > 1:
                        self.yf[n] = (nu - 1) // 2
                    self.zf[n] = 1
                n += 1

    def make_node(self, *inputs):
        inputs = [tt.as_tensor_variable(i) for i in inputs]
        outputs = [i.type() for i in inputs[:-1]]
        return gof.Apply(self, inputs, outputs)

    def infer_shape(self, node, shapes):
        return shapes[:-1]

    def perform(self, node, inputs, outputs):
        x, y, z, bpT = inputs
        bpTpT = bpT * self.base_op.func(x, y, z)
        bx = np.sum(self.xf[None, :] * bpTpT / x[:, None], axis=-1)
        by = np.sum(self.yf[None, :] * bpTpT / y[:, None], axis=-1)
        bz = np.sum(self.zf[None, :] * bpTpT / z[:, None], axis=-1)
        outputs[0][0] = np.reshape(bx, np.shape(inputs[0]))
        outputs[1][0] = np.reshape(by, np.shape(inputs[1]))
        outputs[2][0] = np.reshape(bz, np.shape(inputs[2]))
