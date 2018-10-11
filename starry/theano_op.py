# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["StarryOp"]

import numpy as np

import theano
import theano.tensor as tt

from .kepler import Primary, Secondary, System


_LETTERS = "bcdefghijklmnopqrstuvwxyz"


def assign_y(lmax, obj, y):
    n = 0
    for l in range(1, lmax+1):
        dn = 2*l + 1
        obj[l, :] = y[n:n+dn]
        n += dn


class StarryOp(tt.Op):

    __props__ = (
        "primary_lmax", "primary_u",
        "secondary_lmax", "secondary_u",
    )

    def __init__(self,
                 primary_lmax=2, primary_u=False,
                 secondary_lmax=(2, ), secondary_u=(False, )):
        # Save the primary information
        self.primary_lmax = int(primary_lmax)
        self.primary_u = bool(primary_u)

        self.param_names = []
        if self.primary_u:
            self.param_names.append("A.u")
        else:
            self.param_names.append("A.y")

        # Work out the number and info for the secondaries
        try:
            self.n_secondary = len(secondary_lmax)
        except TypeError:
            self.n_secondary = 1
            secondary_lmax = (secondary_lmax, )
        self.secondary_lmax = tuple(int(l) for l in secondary_lmax)

        try:
            len(secondary_u)
        except TypeError:
            secondary_u = tuple(secondary_u for _ in range(self.n_secondary))
        self.secondary_u = tuple(bool(u) for u in secondary_u)

        if len(self.secondary_u) != len(self.secondary_lmax):
            raise ValueError("the shapes of secondary_lmax and secondary_u "
                             "must match")
        if self.n_secondary > len(_LETTERS):
            raise ValueError("only <= {0} secondaries are allowed"
                             .format(len(_LETTERS)))

        # Save the secondary parameter names
        for u, l in zip(self.secondary_u, _LETTERS):
            if u:
                self.param_names.append("{0}.u".format(l))
            else:
                self.param_names.append("{0}.y".format(l))
            for param in ("L", "r", "a", "porb", "prot", "inc", "ecc", "w",
                          "lambda0"):
                self.param_names.append("{0}.{1}".format(l, param))

        self._grad_op = StarryGradOp(self)

    def make_node(self, *args):
        if len(args) != len(self.param_names) + 1:
            raise ValueError("wrong number of inputs")
        args = [tt.as_tensor_variable(a) for a in args]
        return theano.Apply(self, args, [args[-1].type()])

    def infer_shape(self, node, shapes):
        return shapes[-1],

    def perform(self, node, inputs, outputs):
        system = self.build_system(*(inputs[:-1]))
        system.compute(np.array(inputs[-1]))
        outputs[0][0] = np.array(system.lightcurve)

    def grad(self, inputs, gradients):
        return self._grad_op(*(inputs + gradients))

    def build_system(self, primary_u_or_y, *secondary_args):
        primary = Primary(lmax=self.primary_lmax)
        if self.primary_u:
            primary[:] = primary_u_or_y
        else:
            assign_y(self.primary_lmax, primary, primary_u_or_y)

        secondaries = []
        for i, (u, lmax) in enumerate(zip(self.secondary_u,
                                          self.secondary_lmax)):
            args = secondary_args[10*i:10*(i+1)]
            u_or_y, L, r, a, porb, prot, inc, ecc, w, lambda0 = args

            secondary = Secondary(lmax=lmax)
            if u:
                secondary[:] = u_or_y
            else:
                assign_y(lmax, secondary, u_or_y)

            secondary.L = L
            secondary.r = r
            secondary.a = a
            secondary.porb = porb
            secondary.prot = prot
            secondary.inc = inc
            secondary.ecc = ecc
            secondary.w = w
            secondary.lambda0 = lambda0

            secondaries.append(secondary)

        return System(primary, *secondaries)


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
        system = self.base_op.build_system(*(inputs[:-2]))
        system.compute(np.array(inputs[-2]), gradient=True)
        grads = system.gradient

        # The gradients with respect to the main parameters
        for i, param in enumerate(self.base_op.param_names):
            outputs[i][0] = np.array(np.sum(np.array(grads[param]) *
                                            np.array(inputs[-1]), axis=-1))

        # The gradient with respect to time
        outputs[-1][0] = np.array(grads["time"]) * np.array(inputs[-1])
