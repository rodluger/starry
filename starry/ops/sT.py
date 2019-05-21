# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["STOp"]

import sys
import pkg_resources

import theano
from theano import gof
import theano.tensor as tt


class STOp(gof.COp):

    params_type = gof.ParamsType(
        ydeg=theano.scalar.int32,
        udeg=theano.scalar.int32,
        fdeg=theano.scalar.int32,
    )
    __props__ = ("ydeg", "udeg", "fdeg")
    func_file = "./sT.cc"
    func_name = "APPLY_SPECIFIC(sT)"

    def __init__(self, ydeg=2, udeg=0, fdeg=0, **kwargs):
        self.ydeg = int(ydeg)
        self.udeg = int(udeg)
        self.fdeg = int(fdeg)

        deg = self.ydeg + self.udeg + self.fdeg
        self.N = (deg + 1)**2

        super(STOp, self).__init__(self.func_file, self.func_name)

    def c_code_cache_version(self):
        return ()
        # if "dev" in __version__:
        #     return ()
        # return tuple(map(int, __version__.split(".")))

    def c_headers(self, compiler):
        return ["theano_interface.h", "ops.h", "vector"]

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
            tt.as_tensor_variable(r),
        ]
        out_args = [(tt.shape_padright(in_args[0]) + tt.zeros(self.N)).type()]
        return gof.Apply(self, in_args, out_args)

    def infer_shape(self, node, in_shapes):
        out_shape = list(in_shapes[0])
        out_shape.append(self.N)
        return [out_shape]

    def grad(self, inputs, gradients):
        assert 0
        # c, r, x, xt, xtt, y, yt, ytt, z, zt, dt = inputs
        # f, dfdcl, dfdr, dfdx, dfdxt, dfdxtt, dfdy, dfdyt, dfdytt, neval \
        #     = self(*inputs)
        # bf = gradients[0]
        # for i, g in enumerate(gradients[1:]):
        #     if not isinstance(g.type, theano.gradient.DisconnectedType):
        #         raise ValueError("can't propagate gradients wrt parameter {0}"
        #                          .format(i+1))
        # bc = tt.sum(tt.reshape(bf, (1, bf.size)) *
        #             tt.reshape(dfdcl, (c.size, bf.size)), axis=-1)
        # br = bf * dfdr
        # bx = bf * dfdx
        # bxt = bf * dfdxt
        # bxtt = bf * dfdxtt
        # by = bf * dfdy
        # byt = bf * dfdyt
        # bytt = bf * dfdytt
        # return (
        #     bc, br, bx, bxt, bxtt, by, byt, bytt,
        #     tt.zeros_like(z), tt.zeros_like(zt), tt.zeros_like(dt))

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return eval_points
        return self.grad(inputs, eval_points)

