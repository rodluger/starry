# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np
from theano import gof
import theano.tensor as tt

__all__ = ["YlmFluxOp", "YlmXOp"]


class YlmFluxOp(tt.Op):
    def __init__(self):
        raise NotImplementedError("TODO!")

class YlmXOp(tt.Op):

    def __init__(self, map):
        self.map = map
        self._grad_op = YlmXGradientOp(self)

    def make_node(self, *inputs):
        inputs = [tt.as_tensor_variable(i) for i in inputs]
        outputs = [tt.TensorType(inputs[-1].dtype, (False, False))()]
        return gof.Apply(self, inputs, outputs)

    def infer_shape(self, node, shapes):
        return [shapes[-1] + (tt.as_tensor(self.map.Ny),)]

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return eval_points
        return self.grad(inputs, eval_points)

    def perform(self, node, inputs, outputs):
        u, f, inc, obl, theta, xo, yo, zo, ro = inputs
        if self.map.udeg:
            self.map[1:] = u
        if self.map.fdeg:
            self.map.filter[:, :] = f
        self.map.inc = inc
        self.map.obl = obl
        outputs[0][0] = np.array(self.map._X(theta, xo, yo, zo, ro))

    def grad(self, inputs, gradients):
        return self._grad_op(*(inputs + gradients))


class YlmXGradientOp(tt.Op):

    def __init__(self, base_op):
        self.base_op = base_op

    def make_node(self, *inputs):
        inputs = [tt.as_tensor_variable(i) for i in inputs]
        outputs = [i.type() for i in inputs[:8]]
        return gof.Apply(self, inputs, outputs)

    def infer_shape(self, node, shapes):
        return shapes[:9]

    def perform(self, node, inputs, outputs):
        u, f, inc, obl, theta, xo, yo, zo, ro, bf = inputs
        if self.base_op.map.udeg:
            self.base_op.map[1:] = u
        if self.map.fdeg:
            self.base_op.map.filter[:, :] = f
        self.base_op.map.inc = inc
        self.base_op.map.obl = obl

        # Compute
        btheta, bxo, byo, bro, bu, bf, binc, bobl = \
            self.base_op.map._grad(np.atleast_1d(theta), 
                                   np.atleast_1d(xo), 
                                   np.atleast_1d(yo), 
                                   np.atleast_1d(zo), 
                                   np.atleast_1d(ro),
                                   np.atleast_1d(bf))
        outputs[0][0] = bu
        outputs[1][0] = bf
        outputs[2][0] = binc
        outputs[3][0] = bobl
        outputs[4][0] = btheta
        outputs[5][0] = bxo
        outputs[6][0] = byo
        outputs[7][0] = np.zeros_like(outputs[6][0])
        outputs[8][0] = bro

        # Reshape
        for i in range(9):
            outputs[i][0] = outputs[i][0].reshape(np.shape(inputs[i]))