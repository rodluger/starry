# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np
from theano import gof
import theano.tensor as tt


__all__ = ["tensordotDOp"]


class tensordotDOp(tt.Op):
    def __init__(self, func):
        self.func = func
        self._grad_op = tensordotDGradientOp(self)

    def make_node(self, *inputs):
        inputs = [tt.as_tensor_variable(i) for i in inputs]
        outputs = [tt.TensorType(inputs[0].dtype, (False, False))()]
        return gof.Apply(self, inputs, outputs)

    def infer_shape(self, node, shapes):
        return [[shapes[1][0], shapes[0][-1]]]

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return eval_points
        return self.grad(inputs, eval_points)

    def perform(self, node, inputs, outputs):
        outputs[0][0] = self.func(*inputs)

    def grad(self, inputs, gradients):
        return self._grad_op(*(inputs + gradients))


class tensordotDGradientOp(tt.Op):
    def __init__(self, base_op):
        self.base_op = base_op

    def make_node(self, *inputs):
        inputs = [tt.as_tensor_variable(i) for i in inputs]
        outputs = [i.type() for i in inputs[:-1]]
        return gof.Apply(self, inputs, outputs)

    def infer_shape(self, node, shapes):
        return shapes[:-1]

    def perform(self, node, inputs, outputs):
        bM, bwta = self.base_op.func(*inputs)
        outputs[0][0] = np.reshape(bM, np.shape(inputs[0]))
        outputs[1][0] = np.reshape(bwta, np.shape(inputs[1]))
