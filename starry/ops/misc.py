# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np
from theano import gof
import theano.tensor as tt
import theano.sparse as ts

__all__ = ["spotYlmOp"]


class spotYlmOp(tt.Op):

    def __init__(self, func, ydeg, nw):
        self.func = func
        self.Ny = (ydeg + 1) ** 2
        self.nw = nw

    def make_node(self, *inputs):
        inputs = [tt.as_tensor_variable(i) for i in inputs]
        if self.nw is None:
            outputs = [tt.TensorType(inputs[0].dtype, (False,))()]
        else:
            outputs = [tt.TensorType(inputs[0].dtype, (False, False))()]
        return gof.Apply(self, inputs, outputs)

    def infer_shape(self, node, shapes):
        if self.nw is None:
            return [(self.Ny,)]
        else:
            return [(self.Ny, self.nw)]

    def perform(self, node, inputs, outputs):
        outputs[0][0] = self.func(*inputs)
        if self.nw is None:
            outputs[0][0] = np.reshape(outputs[0][0], -1)