# -*- coding: utf-8 -*-
from ...compat import Apply
import numpy as np
import theano.tensor as tt
import theano.sparse as ts
import os

# C extensions are not installed on RTD
if os.getenv("READTHEDOCS") == "True":
    STARRY_OREN_NAYAR_DEG = 5
else:
    from ..._c_ops import STARRY_OREN_NAYAR_DEG


__all__ = ["FOp", "OrenNayarOp"]


class FOp(tt.Op):
    def __init__(self, func, N, Ny):
        self.func = func
        self.N = N
        self.Ny = Ny
        self._grad_op = FGradientOp(self)

    def make_node(self, *inputs):
        inputs = [tt.as_tensor_variable(i) for i in inputs]
        outputs = [tt.TensorType(inputs[-1].dtype, (False, False))()]
        return Apply(self, inputs, outputs)

    def infer_shape(self, *args):
        return [(self.N, self.Ny)]

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return eval_points
        return self.grad(inputs, eval_points)

    def perform(self, node, inputs, outputs):
        outputs[0][0] = self.func(*inputs)

    def grad(self, inputs, gradients):
        return self._grad_op(*(inputs + gradients))


class FGradientOp(tt.Op):
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
        bu, bf = self.base_op.func(*inputs)
        outputs[0][0] = np.reshape(bu, np.shape(inputs[0]))
        outputs[1][0] = np.reshape(bf, np.shape(inputs[1]))


class OrenNayarOp(tt.Op):
    def __init__(self, func):
        self.func = func

    def make_node(self, *inputs):
        inputs = [tt.as_tensor_variable(i) for i in inputs]
        outputs = [tt.TensorType(inputs[-1].dtype, (False, False))()]
        return Apply(self, inputs, outputs)

    def infer_shape(self, *args):
        shapes = args[-1]
        return [((STARRY_OREN_NAYAR_DEG + 1) ** 2, shapes[0][0])]

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return eval_points
        return self.grad(inputs, eval_points)

    def perform(self, node, inputs, outputs):
        outputs[0][0] = self.func(*inputs)
