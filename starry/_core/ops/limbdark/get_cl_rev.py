# -*- coding: utf-8 -*-
from ....compat import Apply, pt
from .base_op import LimbDarkBaseOp


__all__ = ["GetClRevOp"]


class GetClRevOp(LimbDarkBaseOp):

    __props__ = ()
    func_file = "./get_cl_rev.cc"
    func_name = "APPLY_SPECIFIC(get_cl_rev)"

    def make_node(self, bc):
        return Apply(self, [pt.as_tensor_variable(bc)], [bc.type()])

    def infer_shape(self, *args):
        shapes = args[-1]
        return (shapes[0],)
