# -*- coding: utf-8 -*-
from __future__ import division, print_function
from theano import gof
import theano.tensor as tt
from .base_op import BaseOp
from .sT_rev import sTRevOp


__all__ = ["sTOp"]


class sTOp(BaseOp):
    """

    """

    func_file = "./sT.cc"
    func_name = "APPLY_SPECIFIC(sT)"

    def __init__(self, *args):
        super(sTOp, self).__init__(*args)
        self.grad_op = sTRevOp(*args)

    def make_node(self, b, r):
        in_args = [
            tt.as_tensor_variable(b),
            tt.as_tensor_variable(r)
        ]
        out_args = [
            tt.matrix().type()
        ]
        return gof.Apply(self, in_args, out_args)

    def infer_shape(self, node, in_shapes):
        out_shape = list(in_shapes[0])
        out_shape.append(self.N)
        return [out_shape]

    def grad(self, inputs, gradients):
        return self.grad_op(*(inputs + gradients))

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return eval_points
        return self.grad(inputs, eval_points)