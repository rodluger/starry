# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np
from theano import gof
import theano.tensor as tt

__all__ = ["pT"]


class pT(tt.Op):
    """
    Gradient not implemented!

    """
    def __init__(self, func, N):
        self.func = func
        self.N = N

    def make_node(self, *inputs):
        inputs = [tt.as_tensor_variable(i) for i in inputs]
        outputs = [tt.TensorType(inputs[-1].dtype, (False, False))()]
        return gof.Apply(self, inputs, outputs)

    def infer_shape(self, node, shapes):
        return [shapes[0] + (tt.as_tensor(self.N),)]

    def perform(self, node, inputs, outputs):
        outputs[0][0] = self.func(*inputs)
