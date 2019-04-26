# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np
from theano import gof
import theano.tensor as tt

__all__ = ["LinearOp"]


class LinearOp(tt.Op):

    def __init__(self, map):
        self.map = map
        self._grad_op = LinearGradientOp(self)
        self.occultation = True

    def make_node(self, *inputs):
        inputs = [tt.as_tensor_variable(i) for i in inputs]
        outputs = [tt.TensorType(inputs[-1].dtype, (False, False))()]
        # USED TO BE: outputs = [inputs[-1].type()]
        return gof.Apply(self, inputs, outputs)

    def infer_shape(self, node, shapes):
        return [shapes[-1] + (tt.as_tensor(self.map.Ny),)]
        # USED TO BE: return shapes[-1],

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return eval_points
        return self.grad(inputs, eval_points)

    def perform(self, node, inputs, outputs):
        u, inc, obl, theta, xo, yo, zo, ro = inputs
        if self.map.udeg:
            self.map[1:] = u
        self.map.inc = inc
        self.map.obl = obl
        outputs[0][0] = self.map.linear_flux_model(
            theta=theta, xo=xo, yo=yo, zo=zo, ro=ro)

    def grad(self, inputs, gradients):
        return self._grad_op(*(inputs + gradients))


class LinearGradientOp(tt.Op):

    def __init__(self, base_op):
        self.base_op = base_op

    def make_node(self, *inputs):
        inputs = [tt.as_tensor_variable(i) for i in inputs]
        outputs = [i.type() for i in inputs[:8]]
        return gof.Apply(self, inputs, outputs)

    def infer_shape(self, node, shapes):
        return shapes[:8]

    def perform(self, node, inputs, outputs):
        u, inc, obl, theta, xo, yo, zo, ro, bf = inputs
        if self.base_op.map.udeg:
            self.base_op.map[1:] = u
        self.base_op.map.inc = inc
        self.base_op.map.obl = obl
        _, grad = self.base_op.map.linear_flux_model(
            theta=theta, xo=xo, yo=yo, zo=zo, ro=ro, gradient=True)

        # Limb darkening gradient
        outputs[0][0] = np.array(np.sum(grad["u"] * bf, axis=(1, 2)))

        # Orientation gradients
        outputs[1][0] = np.atleast_1d(np.array(np.sum(grad["inc"] * bf)))
        outputs[2][0] = np.atleast_1d(np.array(np.sum(grad["obl"] * bf)))
        outputs[3][0] = np.array(np.sum(grad["theta"] * bf, axis=-1))

        # Occultation gradients
        if self.occultation:
            outputs[4][0] = np.array(np.sum(grad["xo"] * bf, axis=-1))
            outputs[5][0] = np.array(np.sum(grad["yo"] * bf, axis=-1))
            outputs[6][0] = np.zeros_like(outputs[5][0])
            outputs[7][0] = np.array(np.sum(grad["ro"] * bf, axis=-1))
        else:
            outputs[4][0] = np.empty_like(inputs[4])
            outputs[5][0] = np.empty_like(inputs[5])
            outputs[6][0] = np.empty_like(inputs[6])
            outputs[7][0] = np.empty_like(inputs[7])

        # Reshape
        for i in range(8):
            outputs[i][0] = outputs[i][0].reshape(np.shape(inputs[i]))