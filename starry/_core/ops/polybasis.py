# -*- coding: utf-8 -*-
from ...compat import Apply
import numpy as np
import theano
import theano.tensor as tt

__all__ = ["pTOp"]


class pTOp(tt.Op):
    def __init__(self, func, deg):
        self.func = func
        self.deg = deg
        self.N = (deg + 1) ** 2
        self._grad_op = pTGradientOp(self)

    def make_node(self, *inputs):
        inputs = [tt.as_tensor_variable(i) for i in inputs]
        outputs = [tt.TensorType(inputs[0].dtype, (False, False))()]
        return Apply(self, inputs, outputs)

    def infer_shape(self, *args):
        shapes = args[-1]
        return [[shapes[0][0], self.N]]

    def perform(self, node, inputs, outputs):
        outputs[0][0] = self.func(self.deg, *inputs)

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
        return Apply(self, inputs, outputs)

    def infer_shape(self, *args):
        shapes = args[-1]
        return shapes[:-1]

    def perform(self, node, inputs, outputs):
        x, y, z, bpT = inputs

        # TODO: When any of the coords are zero, there's a div
        # by zero below. This hack fixes the issue. We should
        # think of a better way of doing this!
        tol = 1e-8
        x[np.abs(x) < tol] = tol
        y[np.abs(y) < tol] = tol
        z[np.abs(z) < tol] = tol

        bpTpT = bpT * self.base_op.func(self.base_op.deg, x, y, z)
        bx = np.nansum(self.xf[None, :] * bpTpT / x[:, None], axis=-1)
        by = np.nansum(self.yf[None, :] * bpTpT / y[:, None], axis=-1)
        bz = np.nansum(self.zf[None, :] * bpTpT / z[:, None], axis=-1)
        outputs[0][0] = np.reshape(bx, np.shape(inputs[0]))
        outputs[1][0] = np.reshape(by, np.shape(inputs[1]))
        outputs[2][0] = np.reshape(bz, np.shape(inputs[2]))
