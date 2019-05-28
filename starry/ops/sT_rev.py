# -*- coding: utf-8 -*-
from __future__ import division, print_function
import sys
import pkg_resources
import theano
from theano import gof
import theano.tensor as tt
from theano.tests import unittest_tools as utt
from theano.tensor.extra_ops import CpuContiguous
from .. import __version__

__all__ = ["sTRevOp"]


def as_contiguous_variable(x):
    """Make `x` C-contiguous."""
    return CpuContiguous()(tt.as_tensor_variable(x))


class sTRevOp(gof.COp):
    """

    """

    params_type = gof.ParamsType(
        deg=theano.scalar.int32
    )
    __props__ = ("deg",)
    func_file = "./sT_rev.cc"
    func_name = "APPLY_SPECIFIC(sT_rev)"

    def __init__(self, deg, **kwargs):
        self.deg = int(deg)
        self.N = (deg + 1) ** 2
        super(sTRevOp, self).__init__(self.func_file, self.func_name)

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