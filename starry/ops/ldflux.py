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
        outputs = [inputs[-1].type()]
        return gof.Apply(self, inputs, outputs)

    def infer_shape(self, node, shapes):
        return shapes[-1],

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return eval_points
        return self.grad(inputs, eval_points)

    def perform(self, node, inputs, outputs):
        u, b, zo, ro = inputs
        self.map[1:] = u
        outputs[0][0] = np.array(self.map.flux(b=b, zo=zo, ro=ro))

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
        self.base_op.map[1:] = u
        _, grad = self.base_op.map.flux(b=b, zo=zo, ro=ro, gradient=True)

        # Limb darkening gradient
        outputs[0][0] = np.array(np.sum(grad["u"] * bf, axis=-1))

        # Orbital gradients
        outputs[1][0] = np.array(grad["b"] * bf)
        outputs[2][0] = np.zeros_like(outputs[1][0])

        # Radius gradient
        outputs[3][0] = np.array(grad["ro"] * bf)

        # Reshape
        for i in range(4):
            outputs[i][0] = outputs[i][0].reshape(np.shape(inputs[i]))