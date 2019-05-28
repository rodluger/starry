# -*- coding: utf-8 -*-
from __future__ import division, print_function
import sys
import pkg_resources
import theano
from theano import gof
import theano.tensor as tt
from theano.tests import unittest_tools as utt
from .. import __version__
from .sT_rev import sTRevOp


__all__ = ["sTOp"]


class sTOp(gof.COp):
    """

    """

    params_type = gof.ParamsType(
        deg=theano.scalar.int32
    )
    __props__ = ("deg",)
    func_file = "./sT.cc"
    func_name = "APPLY_SPECIFIC(sT)"

    def __init__(self, deg, **kwargs):
        self.deg = int(deg)
        self.N = (deg + 1) ** 2
        self.grad_op = sTRevOp(deg)
        super(sTOp, self).__init__(self.func_file, self.func_name)

    def c_code_cache_version(self):
        if "dev" in __version__:
            return ()
        return tuple(map(int, __version__.split(".")))

    def c_headers(self, compiler):
        return ["theano_interface.h", "vector"]

    def c_header_dirs(self, compiler):
        return [
            pkg_resources.resource_filename("starry", "include"),
            pkg_resources.resource_filename("starry", "lib/eigen_3.3.5"),
        ]

    def c_compile_args(self, compiler):
        opts = ["-std=c++11", "-O2", "-DNDEBUG"]
        if sys.platform == "darwin":
            opts += ["-stdlib=libc++", "-mmacosx-version-min=10.7"]
        return opts

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