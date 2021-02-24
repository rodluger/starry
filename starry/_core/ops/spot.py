# -*- coding: utf-8 -*-
from ...compat import Apply
import numpy as np
import theano
import theano.tensor as tt

__all__ = ["spotYlmOp"]


class spotYlmOp(tt.Op):
    def __init__(self, func, ydeg, nw):
        self.func = func
        self._grad_op = spotYlmGradientOp(self)
        self.Ny = (ydeg + 1) ** 2
        self.nw = nw

    def make_node(self, *inputs):
        inputs = [tt.as_tensor_variable(i) for i in inputs]
        if self.nw is None:
            outputs = [tt.TensorType(inputs[0].dtype, (False,))()]
        else:
            outputs = [tt.TensorType(inputs[0].dtype, (False, False))()]
        return Apply(self, inputs, outputs)

    def infer_shape(self, *args):
        if self.nw is None:
            return [(self.Ny,)]
        else:
            return [(self.Ny, self.nw)]

    def perform(self, node, inputs, outputs):
        outputs[0][0] = self.func(*inputs)
        if self.nw is None:
            outputs[0][0] = np.reshape(outputs[0][0], -1)

    def grad(self, inputs, gradients):
        return self._grad_op(*(inputs + gradients))


class spotYlmGradientOp(tt.Op):
    def __init__(self, base_op):
        self.base_op = base_op

    def make_node(self, *inputs):
        inputs = [tt.as_tensor_variable(i) for i in inputs]
        outputs = [i.type() for i in inputs[:-1]]
        return Apply(self, inputs, outputs)

    def infer_shape(self, *args):
        shapes = args[-1]
        return shapes[:-1]

    def perform(self, node, inputs, outputs):
        bamp, bsigma, blat, blon = self.base_op.func(*inputs)
        outputs[0][0] = np.reshape(bamp, np.shape(inputs[0]))
        outputs[1][0] = np.reshape(bsigma, np.shape(inputs[1]))
        outputs[2][0] = np.reshape(blat, np.shape(inputs[2]))
        outputs[3][0] = np.reshape(blon, np.shape(inputs[3]))
