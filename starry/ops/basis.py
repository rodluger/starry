# -*- coding: utf-8 -*-
from __future__ import division, print_function
from theano import gof
import theano.tensor as tt
from .base_op import BaseOp


__all__ = ["ChangeOfBasisOp"]


class ChangeOfBasisOp(BaseOp):
    """

    """

    func_file = "./basis.cc"
    func_name = "APPLY_SPECIFIC(basis)"

    def __init__(self, *args):
        super(ChangeOfBasisOp, self).__init__(*args)

    def make_node(self):
        in_args = []
        out_args = [
            tt.vector().type(),
            tt.vector().type(),
            tt.matrix().type(),
            tt.matrix().type(),
            tt.matrix().type()
        ]
        return gof.Apply(self, in_args, out_args)

    def infer_shape(self, node, in_shapes):
        out_shape = [[self.N], [self.Ny], [self.N, self.N], 
                     [self.Ny, self.Ny], [self.N, self.N]]
        return out_shape