# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np
import theano
import theano.tensor as tt
from .._starry_default_double import Map

__all__ = ["StarryOp"]


class StarryOp(tt.Op):

    __props__ = ("lmax", )

    def __init__(self, lmax=2):
        # Save the primary information
        self.lmax = int(lmax)
        self.param_names = ["y", "u", "theta", "xo", "yo", "ro", "zo"]

        # Pre-initialize the Map object
        self.map = Map(lmax=self.lmax)

        # Set up the gradient operation
        self._grad_op = StarryGradOp(self)

    def make_node(self, *args):
        if len(args) != len(self.param_names):
            raise ValueError("Incorrect number of inputs.")
        args = [tt.as_tensor_variable(a) for a in args]
        return theano.Apply(self, args, [args[-1].type()])

    def infer_shape(self, node, shapes):
        return shapes[-1],

    def perform(self, node, inputs, outputs):
        y, u, theta, xo, yo, ro, zo = inputs
        self.map[:, :] = y
        self.map[:] = u
        ro_ = tt.switch(tt.lt(zo, 0.0), ro, 0.0).eval()
        outputs[0][0] = self.map.flux(theta=theta, xo=xo, yo=yo, ro=ro_)

    def grad(self, inputs, gradients):
        return self._grad_op(*(inputs + gradients))

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return eval_points
        return self.grad(inputs, eval_points)


class StarryGradOp(tt.Op):

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
        y, u, theta, xo, yo, ro, zo, DDf = inputs
        self.base_op.map[:, :] = y
        self.base_op.map[:] = u

        ro_ = tt.switch(tt.lt(zo, 0.0), ro, 0.0).eval()
        _, grads = self.base_op.map.flux(theta=theta, xo=xo, yo=yo, 
                                         ro=ro_, gradient=True)

        # The gradients with respect to the static parameters
        for i, param in enumerate(["y", "u"]):
            shape = list(inputs[i].shape) + list(DDf.shape)
            outputs[i][0] = np.array(np.sum(grads.get(param, np.zeros(shape)) * DDf, axis=-1))

        # The gradients with respect to the time-dependent parameters
        for i, param in enumerate(["theta", "xo", "yo", "ro"]):
            outputs[i + 2][0] = np.array(grads.get(param, 0.0) * DDf)