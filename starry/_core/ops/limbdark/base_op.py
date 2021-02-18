# -*- coding: utf-8 -*-
from ....compat import COp
from ....starry_version import __version__
import sys
import pkg_resources


__all__ = ["LimbDarkBaseOp"]


class LimbDarkBaseOp(COp):

    __props__ = ()
    func_file = None
    func_name = None

    def __init__(self):
        super(LimbDarkBaseOp, self).__init__(self.func_file, self.func_name)

    def c_code_cache_version(self, *args, **kwargs):
        if "dev" in __version__:
            return ()
        return tuple(map(int, __version__.split(".")))

    def c_headers(self, *args, **kwargs):
        return [
            "theano_helpers.h",
            "ellip.h",
            "limbdark.h",
            "utils.h",
            "vector",
        ]

    def c_header_dirs(self, *args, **kwargs):
        dirs = [
            pkg_resources.resource_filename("starry", "_core/ops/lib/include")
        ]
        dirs += [
            pkg_resources.resource_filename(
                "starry", "_core/ops/lib/vendor/eigen_3.3.5"
            )
        ]
        return dirs

    def c_compile_args(self, *args, **kwargs):
        opts = ["-std=c++11", "-O2", "-DNDEBUG"]
        if sys.platform == "darwin":
            opts += ["-stdlib=libc++", "-mmacosx-version-min=10.7"]
        return opts

    def perform(self, *args):
        raise NotImplementedError("Only C op is implemented")
