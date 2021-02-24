# -*- coding: utf-8 -*-
from ...compat import Apply
import numpy as np
import theano.tensor as tt

__all__ = ["CheckBoundsOp", "RaiseValueErrorOp", "RaiseValueErrorIfOp"]


class CheckBoundsOp(tt.Op):
    """

    """

    def __init__(self, lower=-np.inf, upper=np.inf, name=None):
        self.lower = lower
        self.upper = upper
        if name is None:
            self.name = "parameter"
        else:
            self.name = name

    def make_node(self, *inputs):
        inputs = [tt.as_tensor_variable(inputs[0])]
        outputs = [inputs[0].type()]
        return Apply(self, inputs, outputs)

    def infer_shape(self, *args):
        shapes = args[-1]
        return [shapes[0]]

    def perform(self, node, inputs, outputs):
        outputs[0][0] = inputs[0]
        if np.any((inputs[0] < self.lower) | (inputs[0] > self.upper)):
            low = np.where((inputs[0] < self.lower))[0]
            high = np.where((inputs[0] > self.upper))[0]
            if len(low):
                value = inputs[0][low[0]]
                sign = "<"
                bound = self.lower
            else:
                value = inputs[0][high[0]]
                sign = ">"
                bound = self.upper
            raise ValueError(
                "%s out of bounds: %f %s %f" % (self.name, value, sign, bound)
            )


class RaiseValueErrorIfOp(tt.Op):
    """

    """

    def __init__(self, message=None):
        self.message = message

    def make_node(self, *inputs):
        condition = inputs
        inputs = [tt.as_tensor_variable(condition)]
        outputs = [tt.TensorType(tt.config.floatX, ())()]
        return gof.Apply(self, inputs, outputs)

    def infer_shape(self, *args):
        return [()]

    def perform(self, node, inputs, outputs):
        outputs[0][0] = np.array(0.0)
        if inputs[0]:
            raise ValueError(self.message)

    def grad(self, inputs, gradients):
        return [inputs[0] * 0.0]


def RaiseValueErrorOp(msg, shape=()):
    return tt.zeros(shape) * RaiseValueErrorIfOp(msg)(True)
