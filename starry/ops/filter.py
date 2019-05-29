# -*- coding: utf-8 -*-
from __future__ import division, print_function
from theano import gof
import theano.tensor as tt
from .base_op import BaseOp
from .filter_rev import FilterRevOp


__all__ = ["FilterOp"]


class FilterOp(BaseOp):
    """

    """

    func_file = "./filter.cc"
    func_name = "APPLY_SPECIFIC(filter)"

    def __init__(self, *args):
        super(FilterOp, self).__init__(*args)
        self.grad_op = FilterRevOp(*args)

    def make_node(self, u, f):
        in_args = [
            tt.as_tensor_variable(u),
            tt.as_tensor_variable(f)
        ]
        out_args = [
            tt.matrix().type()
        ]
        return gof.Apply(self, in_args, out_args)

    def infer_shape(self, node, in_shapes):
        out_shape = [[self.N, self.Ny]]
        return out_shape

    def grad(self, inputs, gradients):
        return self.grad_op(*(inputs + gradients))

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return eval_points
        return self.grad(inputs, eval_points)