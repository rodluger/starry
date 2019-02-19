# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np
import theano
import theano.tensor as tt
from .._starry_default_double import Map

__all__ = ["DefaultYlmOp"]


class DefaultYlmOp(tt.Op):

    __props__ = ("lmax", )

    def __init__(self, lmax=2):
        # Save the primary information
        self.lmax = int(lmax)

        # Pre-initialize the Map object
        self.map = Map(lmax=self.lmax)

        # Set up the gradient operation
        self._grad_op = DefaultYlmGradOp(self)

    def make_node(self, *args):
        if len(args) != 6:
            raise ValueError("Incorrect number of inputs.")
        args = [tt.as_tensor_variable(a) for a in args]
        out_args = [args[-1].type() for i in range(7)]
        return theano.Apply(self, args, out_args)

    def infer_shape(self, node, shapes):
        return shapes[-1],

    def perform(self, node, inputs, outputs):
        y, theta, xo, yo, zo, ro = inputs
        self.map[:, :] = y
        # HACK: nudge at least one ylm away from zero
        # to force starry to compute all derivatives
        if (len(y) > 2) and (y[2] == 0):
            self.map[1, 0] = 1.e-15
        outputs[0][0] = self.map.flux(theta=theta, xo=xo, yo=yo, zo=zo, ro=ro)
            
    def grad(self, inputs, gradients):
        return self._grad_op(*(inputs + gradients))

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return eval_points
        return self.grad(inputs, eval_points)


class DefaultYlmGradOp(tt.Op):

    __props__ = ("base_op", )

    def __init__(self, base_op):
        self.base_op = base_op

    def make_node(self, *args):
        if len(args) != len(self.base_op.param_names) + 1:
            raise ValueError("Incorrect number of inputs.")
        args = [tt.as_tensor_variable(a) for a in args]
        return theano.Apply(self, args, [a.type() for a in args[:-1]])

    def infer_shape(self, node, shapes):
        return shapes[:-1]

    def perform(self, node, inputs, outputs):
        y, theta, xo, yo, zo, ro, DDf = inputs
        self.base_op.map[:, :] = y
        # HACK: nudge at least one ylm away from zero
        # to force starry to compute all derivatives
        if (len(y) > 2) and (y[2] == 0):
            self.base_op.map[1, 0] = 1.e-15
        _, grads = self.base_op.map.flux(theta=theta, xo=xo, yo=yo, zo=zo,
                                         ro=ro, gradient=True)

        # The map gradient
        shape = list(inputs[0].shape) + list(DDf.shape)
        outputs[0][0] = np.array(np.sum(grads.get("y", np.zeros(shape)) * DDf, axis=-1))

        # The gradients with respect to the time-dependent parameters
        for i, param in enumerate(["theta", "xo", "yo", "zo"]):
            outputs[i + 1][0] = np.array(grads.get(param, 0.0) * DDf)
        
        # The radius gradient
        shape = list(inputs[5].shape) + list(DDf.shape)
        outputs[5][0] = np.atleast_1d(np.array(np.sum(grads.get("ro", np.zeros(shape)) * DDf, axis=-1)))