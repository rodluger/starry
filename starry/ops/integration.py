# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np
from theano import gof
import theano.tensor as tt

__all__ = ["sT"]


class sT(tt.Op):

    def __init__(self, c_ops):
        self.c_ops = c_ops
        self._grad_op = sTGradient(self)

    def make_node(self, *inputs):
        inputs = [tt.as_tensor_variable(i) for i in inputs]
        outputs = [tt.TensorType(inputs[-1].dtype, (False, False))()]
        return gof.Apply(self, inputs, outputs)

    def infer_shape(self, node, shapes):
        return [shapes[0] + (tt.as_tensor(self.c_ops.N),)]

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return eval_points
        return self.grad(inputs, eval_points)

    def perform(self, node, inputs, outputs):
        outputs[0][0] = self.c_ops.sT(*inputs)

    def grad(self, inputs, gradients):
        return self._grad_op(*(inputs + gradients))


class sTGradient(tt.Op):

    def __init__(self, base_op):
        self.base_op = base_op

    def make_node(self, *inputs):
        inputs = [tt.as_tensor_variable(i) for i in inputs]
        outputs = [i.type() for i in inputs[:-1]]
        return gof.Apply(self, inputs, outputs)

    def infer_shape(self, node, shapes):
        return shapes[:-1]

    def perform(self, node, inputs, outputs):
        bb, br = self.base_op.c_ops.sT(*inputs)
        outputs[0][0] = np.reshape(bb, np.shape(inputs[0]))
        outputs[1][0] = np.reshape(br, np.shape(inputs[1]))
