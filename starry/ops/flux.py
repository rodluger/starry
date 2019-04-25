# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np
from theano import gof
import theano.tensor as tt

__all__ = ["FluxOp"]


class FluxOp(tt.Op):

    def __init__(self, map):
        self.map = map
        self._grad_op = FluxGradientOp(self)

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
        y, u, inc, obl, theta, xo, yo, zo, ro = inputs
        if self.map.ydeg:
            self.map[1:, :] = y
        if self.map.udeg:
            self.map[1:] = u
        self.map.inc = inc
        self.map.obl = obl
        outputs[0][0] = self.map.flux(theta=theta, xo=xo, yo=yo, zo=zo, ro=ro)

    def grad(self, inputs, gradients):
        return self._grad_op(*(inputs + gradients))


class FluxGradientOp(tt.Op):

    def __init__(self, base_op):
        self.base_op = base_op

    def make_node(self, *inputs):
        inputs = [tt.as_tensor_variable(i) for i in inputs]
        outputs = [i.type() for i in inputs[:9]]
        return gof.Apply(self, inputs, outputs)

    def infer_shape(self, node, shapes):
        return shapes[:9]

    def perform(self, node, inputs, outputs):
        y, u, inc, obl, theta, xo, yo, zo, ro, bf = inputs
        if self.base_op.map.ydeg:
            self.base_op.map[1:, :] = y
        if self.base_op.map.udeg:
            self.base_op.map[1:] = u
        self.base_op.map.inc = inc
        self.base_op.map.obl = obl
        _, grad = self.base_op.map.flux(theta=theta, xo=xo, yo=yo,
                                        zo=zo, ro=ro, gradient=True)

        # Spherical harmonics gradient
        outputs[0][0] = np.array(np.sum(grad["y"] * bf, axis=-1))

        # Limb darkening gradient
        outputs[1][0] = np.array(np.sum(grad["u"] * bf, axis=-1))

        # Orientation gradients
        outputs[2][0] = np.atleast_1d(np.array(np.sum(grad["inc"] * bf, axis=-1)))
        outputs[3][0] = np.atleast_1d(np.array(np.sum(grad["obl"] * bf, axis=-1)))

        # Orbital gradients
        outputs[4][0] = np.array(grad["theta"] * bf)
        outputs[5][0] = np.array(grad["xo"] * bf)
        outputs[6][0] = np.array(grad["yo"] * bf)
        outputs[7][0] = np.zeros_like(outputs[6][0])

        # Radius gradient
        outputs[8][0] = np.array(grad["ro"] * bf)

        # Reshape
        for i in range(9):
            outputs[i][0] = outputs[i][0].reshape(np.shape(inputs[i]))