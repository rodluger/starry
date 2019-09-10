# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np
from theano import gof
import theano.tensor as tt
import theano.sparse as ts

__all__ = ["spotYlmOp", "pTOp"]


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