# -*- coding: utf-8 -*-
from __future__ import division, print_function
import pkg_resources
import theano
from theano import gof
import sys
from .. import __version__


__all__ = ["BaseOp"]


class BaseOp(gof.COp):
    """

    """

    __props__ = ("ydeg", "Ny", "udeg", "Nu", "fdeg", "Nf", "deg", "N")
    params_type = gof.ParamsType(
        **{prop: theano.scalar.int32 for prop in __props__}
    )
    func_file = None
    func_name = None

    def __init__(self, ydeg, udeg, fdeg):
        self.ydeg = int(ydeg)
        self.Ny = (self.ydeg + 1) ** 2
        self.udeg = int(udeg)
        self.Nu = self.udeg + 1
        self.fdeg = int(fdeg)
        self.Nf = (self.fdeg + 1) ** 2
        self.deg = self.ydeg + self.udeg + self.fdeg
        self.N = (self.deg + 1) ** 2
        super(BaseOp, self).__init__(self.func_file, self.func_name)

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