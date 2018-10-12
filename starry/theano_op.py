# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["StarryOp"]

import numpy as np

import theano
import theano.tensor as tt

from .kepler import Primary, Secondary, System


_LETTERS = "bcdefghijklmnopqrstuvwxyz"


class StarryOp(tt.Op):

    __props__ = ("primary_lmax", "secondary_lmax", )

    def __init__(self, primary_lmax=2, secondary_lmax=(2, )):
        # Save the primary information
        self.primary_lmax = int(primary_lmax)
        self.param_names = ["A.u", "A.y"]

        # Work out the number and info for the secondaries
        try:
            self.n_secondary = len(secondary_lmax)
        except TypeError:
            self.n_secondary = 1
            secondary_lmax = (secondary_lmax, )
        self.secondary_lmax = tuple(int(l) for l in secondary_lmax)
        if self.n_secondary > len(_LETTERS):
            raise ValueError("only <= {0} secondaries are allowed"
                             .format(len(_LETTERS)))

        # Save the secondary parameter names
        for l in _LETTERS[:self.n_secondary]:
            for param in ("u", "y", "L", "r", "a", "porb", "prot", "inc",
                          "ecc", "w", "lambda0"):
                self.param_names.append("{0}.{1}".format(l, param))

        # Pre-initialize the starry objects
        self.primary = Primary(lmax=self.primary_lmax)
        self.secondaries = [Secondary(lmax=lmax)
                            for lmax in self.secondary_lmax]
        self.system = System(self.primary, *self.secondaries)

        # Set up the gradient operation
        self._grad_op = StarryGradOp(self)

    def make_node(self, *args):
        if len(args) != len(self.param_names) + 1:
            raise ValueError("wrong number of inputs")
        args = [tt.as_tensor_variable(a) for a in args]
        return theano.Apply(self, args, [args[-1].type()])

    def infer_shape(self, node, shapes):
        return shapes[-1],

    def perform(self, node, inputs, outputs):
        self.build_system(*(inputs[:-1]))
        self.system.compute(np.array(inputs[-1]))
        outputs[0][0] = np.array(self.system.lightcurve)

    def grad(self, inputs, gradients):
        return self._grad_op(*(inputs + gradients))

    def build_system(self, primary_u, primary_y, *secondary_args):
        self.primary[:] = primary_u
        self.primary[1:, :] = primary_y

        for i, secondary in enumerate(self.secondaries):
            args = secondary_args[11*i:11*(i+1)]
            u, y, L, r, a, porb, prot, inc, ecc, w, lambda0 = args

            secondary[:] = u
            secondary[1:, :] = y
            secondary.L = L
            secondary.r = r
            secondary.a = a
            secondary.porb = porb
            secondary.prot = prot
            secondary.inc = inc
            secondary.ecc = ecc
            secondary.w = w
            secondary.lambda0 = lambda0


class StarryGradOp(tt.Op):

    __props__ = ("base_op", )

    def __init__(self, base_op):
        self.base_op = base_op

    def make_node(self, *args):
        if len(args) != len(self.base_op.param_names) + 2:
            raise ValueError("wrong number of inputs")
        args = [tt.as_tensor_variable(a) for a in args]
        return theano.Apply(self, args, [a.type() for a in args[:-1]])

    def infer_shape(self, node, shapes):
        return shapes[:-1]

    def perform(self, node, inputs, outputs):
        self.base_op.build_system(*(inputs[:-2]))
        self.base_op.system.compute(np.array(inputs[-2]), gradient=True)
        grads = self.base_op.system.gradient

        # The gradients with respect to the main parameters
        for i, param in enumerate(self.base_op.param_names):
            outputs[i][0] = np.array(np.sum(grads.get(param, 0.0) * inputs[-1],
                                            axis=-1))

        # The gradient with respect to time
        outputs[-1][0] = np.array(grads.get("time", 0.0) * inputs[-1])
