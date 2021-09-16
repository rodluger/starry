# -*- coding: utf-8 -*-
from ...compat import Apply, Op, tt, theano
import numpy as np

__all__ = ["setMatrixOp"]


def set_matrix(matrix, i, j, vals):
    matrix = np.ascontiguousarray(matrix)
    old_shape = matrix[i, j].shape
    new_shape = np.atleast_2d(vals).shape
    if old_shape == new_shape:
        matrix[i, j] = vals
    elif old_shape == new_shape[::-1]:
        matrix[i, j] = np.atleast_2d(vals).T
    else:
        matrix[i, j] = vals
    return matrix


class setMatrixOp(Op):
    def make_node(self, *inputs):
        inputs = [tt.as_tensor_variable(i) for i in inputs]
        outputs = [inputs[0].type()]
        return Apply(self, inputs, outputs)

    def infer_shape(self, *args):
        shapes = args[-1]
        return [shapes[0]]

    def perform(self, node, inputs, outputs):
        outputs[0][0] = set_matrix(*inputs)

    def grad(self, inputs, gradients):
        return setMatrixGradientOp()(*(inputs + gradients))


class setMatrixGradientOp(Op):
    def make_node(self, *inputs):
        inputs = [tt.as_tensor_variable(i) for i in inputs]
        outputs = [i.type() for i in inputs[:-1]]
        return Apply(self, inputs, outputs)

    def infer_shape(self, *args):
        shapes = args[-1]
        return shapes[:-1]

    def perform(self, node, inputs, outputs):
        # TODO: Is this hard to code up?
        raise NotImplementedError("")
        matrix, i, j, vals, bf = inputs
        outputs[0][0] = None  # ??
        outputs[1][0] = theano.gradient.grad_not_implemented()
        outputs[2][0] = theano.gradient.grad_not_implemented()
        outputs[3][0] = None  # ??
