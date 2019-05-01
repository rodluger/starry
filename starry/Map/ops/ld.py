# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np
from theano import gof
import theano.tensor as tt

__all__ = ["LimbDarkenedOp"]


class LimbDarkenedOp(tt.Op):

    def __init__(self, map):
        self.map = map
        self._grad_op = LimbDarkenedOpGradientOp(self)

    def make_node(self, *inputs):
        inputs = [tt.as_tensor_variable(i) for i in inputs]
        if self.map._spectral:
            outputs = [tt.TensorType(inputs[-1].dtype, (False, False))()]
        else:
            outputs = [inputs[-1].type()]
        return gof.Apply(self, inputs, outputs)

    def infer_shape(self, node, shapes):
        if self.map._spectral:
            return [shapes[-1] + (tt.as_tensor(self.map.nw),)]
        else:
            return shapes[-1],

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return eval_points
        return self.grad(inputs, eval_points)

    def perform(self, node, inputs, outputs):
        u, b, zo, ro = inputs
        if self.map._spectral:
            self.map[1:, :] = u
        else:
            self.map[1:] = u
        outputs[0][0] = self.map._flux(np.atleast_1d(b), 
                                       np.atleast_1d(zo), 
                                       np.atleast_1d(ro))

    def grad(self, inputs, gradients):
        return self._grad_op(*(inputs + gradients))


class LimbDarkenedOpGradientOp(tt.Op):

    def __init__(self, base_op):
        self.base_op = base_op

    def make_node(self, *inputs):
        inputs = [tt.as_tensor_variable(i) for i in inputs]
        outputs = [i.type() for i in inputs[:4]]
        return gof.Apply(self, inputs, outputs)

    def infer_shape(self, node, shapes):
        return shapes[:4]

    def perform(self, node, inputs, outputs):
        u, b, zo, ro, bf = inputs
        if self.base_op.map._spectral:
            self.base_op.map[1:, :] = u
        else:
            self.base_op.map[1:] = u
        bb, bro, bu = self.base_op.map._grad(np.atleast_1d(b), 
                                             np.atleast_1d(zo), 
                                             np.atleast_1d(ro),
                                             np.atleast_1d(bf))

        # Limb darkening gradient
        outputs[0][0] = bu

        # Orbital gradients
        outputs[1][0] = bb
        outputs[2][0] = np.zeros_like(outputs[1][0])

        # Radius gradient
        outputs[3][0] = bro

        # Reshape
        for i in range(4):
            outputs[i][0] = outputs[i][0].reshape(np.shape(inputs[i]))