# -*- coding: utf-8 -*-
from __future__ import division, print_function
from theano import gof
import theano.tensor as tt
from .base_op import BaseOp
from .utils import as_contiguous_variable


__all__ = ["sTRevOp"]


class sTRevOp(BaseOp):
    """

    """

    func_file = "./sT_rev.cc"
    func_name = "APPLY_SPECIFIC(sT_rev)"

    def make_node(self, b, r, bsT):
        in_args = [
            tt.as_tensor_variable(b),
            tt.as_tensor_variable(r),
            as_contiguous_variable(bsT)
        ]
        out_args = [
            in_args[0].type(),
            in_args[1].type()
        ]
        return gof.Apply(self, in_args, out_args)

    def infer_shape(self, node, in_shapes):
        return [in_shapes[0], in_shapes[1]]