# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np
import theano
import theano.tensor as tt
from ..Map import DopplerMap

__all__ = ["DopplerMapOp"]


class DopplerMapOp(tt.Op):

    itypes = [tt.dvector, tt.dvector, tt.dscalar, tt.dscalar, tt.dscalar,
              tt.dscalar, tt.dvector, tt.dvector, tt.dvector, tt.dvector,
              tt.dvector]
    otypes = [theano.tensor.dvector]

    def __init__(self, ydeg=0, udeg=0):
        self.ydeg = ydeg
        self.udeg = udeg
        self.map = DopplerMap(ydeg=self.ydeg, udeg=self.udeg)
        self._grad_op = DopplerMapGradientOp(self)

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return eval_points
        return self.grad(inputs, eval_points)

    def perform(self, node, inputs, outputs):
        y, u, inc, obl, veq, alpha, theta, xo, yo, zo, ro = inputs
        if self.ydeg:
            self.map[1:, :] = y
        if self.udeg:
            self.map[1:] = u
        self.map.inc = inc
        self.map.obl = obl
        self.map.veq = veq
        self.map.alpha = alpha
        outputs[0][0] = self.map.rv(theta=theta, xo=xo, yo=yo, zo=zo, ro=ro)

    def grad(self, inputs, gradients):
        return self._grad_op(*(inputs + gradients))


class DopplerMapGradientOp(tt.Op):

    itypes = [tt.dvector, tt.dvector, tt.dscalar, tt.dscalar, tt.dscalar,
              tt.dscalar, tt.dvector, tt.dvector, tt.dvector, tt.dvector,
              tt.dvector, tt.dvector]
    otypes = [tt.dvector, tt.dvector, tt.dscalar, tt.dscalar, tt.dscalar,
              tt.dscalar, tt.dvector, tt.dvector, tt.dvector, tt.dvector,
              tt.dvector]

    def __init__(self, base_op):
        self.base_op = base_op

    def perform(self, node, inputs, outputs):
        y, u, inc, obl, veq, alpha, theta, xo, yo, zo, ro, bf = inputs
        if self.base_op.ydeg:
            self.base_op.map[1:, :] = y
        if self.base_op.udeg:
            self.base_op.map[1:] = u
        self.base_op.map.inc = inc
        self.base_op.map.obl = obl
        self.base_op.map.veq = veq
        self.base_op.map.alpha = alpha
        _, grad = self.base_op.map.rv(theta=theta, xo=xo, yo=yo, 
                                      zo=zo, ro=ro, gradient=True)
        
        # Spherical harmonics gradient
        outputs[0][0] = np.array(np.sum(grad["y"] * bf, axis=-1))

        # Limb darkening gradient
        outputs[1][0] = np.array(np.sum(grad["u"] * bf, axis=-1))

        # RV field gradients
        outputs[2][0] = np.atleast_1d(np.array(np.sum(grad["inc"] * bf, axis=-1)))
        outputs[3][0] = np.atleast_1d(np.array(np.sum(grad["obl"] * bf, axis=-1)))
        outputs[4][0] = np.atleast_1d(np.array(np.sum(grad["veq"] * bf, axis=-1)))
        outputs[5][0] = np.atleast_1d(np.array(np.sum(grad["alpha"] * bf, axis=-1)))

        # Orbital gradients
        outputs[6][0] = np.array(grad["theta"] * bf)
        outputs[7][0] = np.array(grad["xo"] * bf)
        outputs[8][0] = np.array(grad["yo"] * bf)
        outputs[9][0] = np.zeros_like(outputs[8][0])

        # Radius gradient
        outputs[10][0] = np.atleast_1d(np.array(np.sum(grad["ro"] * bf, axis=-1)))