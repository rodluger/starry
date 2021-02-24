# -*- coding: utf-8 -*-
from ...compat import Apply
import numpy as np
import theano.tensor as tt


__all__ = ["sTOp", "rTReflectedOp", "sTReflectedOp"]


class sTOp(tt.Op):
    def __init__(self, func, N):
        self.func = func
        self.N = N
        self._grad_op = sTGradientOp(self)

    def make_node(self, *inputs):
        inputs = [tt.as_tensor_variable(i) for i in inputs]
        outputs = [tt.TensorType(inputs[-1].dtype, (False, False))()]
        return Apply(self, inputs, outputs)

    def infer_shape(self, *args):
        shapes = args[-1]
        return [shapes[0] + (tt.as_tensor(self.N),)]

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return eval_points
        return self.grad(inputs, eval_points)

    def perform(self, node, inputs, outputs):
        outputs[0][0] = self.func(*inputs)

    def grad(self, inputs, gradients):
        return self._grad_op(*(inputs + gradients))


class sTGradientOp(tt.Op):
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
        bb, br = self.base_op.func(*inputs)
        outputs[0][0] = np.reshape(bb, np.shape(inputs[0]))
        outputs[1][0] = np.reshape(br, np.shape(inputs[1]))


class rTReflectedOp(tt.Op):
    def __init__(self, func, N):
        self.func = func
        self.N = N
        self._grad_op = rTReflectedGradientOp(self)

    def make_node(self, *inputs):
        inputs = [tt.as_tensor_variable(i) for i in inputs]
        outputs = [
            tt.TensorType(inputs[-1].dtype, (False, False))(),
            tt.TensorType(inputs[-1].dtype, (False, False))(),
            tt.TensorType(inputs[-1].dtype, (False, False))(),
        ]
        return Apply(self, inputs, outputs)

    def infer_shape(self, *args):
        shapes = args[-1]
        return [
            shapes[0] + (tt.as_tensor(self.N),),
            shapes[0] + (tt.as_tensor(self.N),),
            shapes[0] + (tt.as_tensor(self.N),),
        ]

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return eval_points
        return self.grad(inputs, eval_points)

    def perform(self, node, inputs, outputs):
        (b, sigr) = inputs
        rT, ddb, ddsigr = self.func(b, sigr)
        outputs[0][0] = rT
        outputs[1][0] = ddb
        outputs[2][0] = ddsigr

    def grad(self, inputs, gradients):
        results = self(*inputs)
        grad = self._grad_op(*(inputs + results + [gradients[0]]))
        return grad


class rTReflectedGradientOp(tt.Op):
    def __init__(self, base_op):
        self.base_op = base_op

    def make_node(self, *inputs):
        inputs = [tt.as_tensor_variable(i) for i in inputs]
        outputs = [i.type() for i in inputs[:2]]
        return Apply(self, inputs, outputs)

    def infer_shape(self, *args):
        shapes = args[-1]
        return shapes[:2]

    def perform(self, node, inputs, outputs):
        b, sigr, rT, ddb, ddsigr, brT = inputs
        bb = (brT * ddb).sum(-1)
        bsigr = (brT * ddsigr).sum()
        outputs[0][0] = np.reshape(bb, np.shape(b))
        outputs[1][0] = np.array(np.reshape(bsigr, np.shape(sigr)))


class sTReflectedOp(tt.Op):
    def __init__(self, func, N):
        self.func = func
        self.N = N
        self._grad_op = sTReflectedGradientOp(self)

    def make_node(self, *inputs):
        inputs = [tt.as_tensor_variable(i) for i in inputs]
        outputs = [
            tt.TensorType(inputs[-1].dtype, (False, False))(),
            tt.TensorType(inputs[-1].dtype, (False, False))(),
            tt.TensorType(inputs[-1].dtype, (False, False))(),
            tt.TensorType(inputs[-1].dtype, (False, False))(),
            tt.TensorType(inputs[-1].dtype, (False, False))(),
            tt.TensorType(inputs[-1].dtype, (False, False))(),
        ]
        return Apply(self, inputs, outputs)

    def infer_shape(self, *args):
        shapes = args[-1]
        return [
            shapes[0] + (tt.as_tensor(self.N),),
            shapes[0] + (tt.as_tensor(self.N),),
            shapes[0] + (tt.as_tensor(self.N),),
            shapes[0] + (tt.as_tensor(self.N),),
            shapes[0] + (tt.as_tensor(self.N),),
            shapes[0] + (tt.as_tensor(self.N),),
        ]

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return eval_points
        return self.grad(inputs, eval_points)

    def perform(self, node, inputs, outputs):
        b, theta, bo, ro, sigr = inputs
        sT, ddb, ddtheta, ddbo, ddro, ddsigr = self.func(
            b, theta, bo, ro, sigr
        )
        outputs[0][0] = sT
        outputs[1][0] = ddb
        outputs[2][0] = ddtheta
        outputs[3][0] = ddbo
        outputs[4][0] = ddro
        outputs[5][0] = ddsigr

    def grad(self, inputs, gradients):
        results = self(*inputs)
        grad = self._grad_op(*(inputs + results + [gradients[0]]))
        return grad


class sTReflectedGradientOp(tt.Op):
    def __init__(self, base_op):
        self.base_op = base_op

    def make_node(self, *inputs):
        inputs = [tt.as_tensor_variable(i) for i in inputs]
        outputs = [i.type() for i in inputs[:5]]
        return Apply(self, inputs, outputs)

    def infer_shape(self, *args):
        shapes = args[-1]
        return shapes[:5]

    def perform(self, node, inputs, outputs):
        (
            b,
            theta,
            bo,
            ro,
            sigr,
            sT,
            ddb,
            ddtheta,
            ddbo,
            ddro,
            ddsigr,
            bsT,
        ) = inputs
        bb = (bsT * ddb).sum(-1)
        btheta = (bsT * ddtheta).sum(-1)
        bbo = (bsT * ddbo).sum(-1)
        bro = (bsT * ddro).sum()
        bsigr = (bsT * ddsigr).sum()
        outputs[0][0] = np.reshape(bb, np.shape(b))
        outputs[1][0] = np.reshape(btheta, np.shape(theta))
        outputs[2][0] = np.reshape(bbo, np.shape(bo))
        outputs[3][0] = np.array(np.reshape(bro, np.shape(ro)))
        outputs[4][0] = np.array(np.reshape(bsigr, np.shape(sigr)))
