"""Install script for `starry`."""
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import warnings
import setuptools
import subprocess
import sys
import os
import glob

# Debug mode?
debug = bool(int(os.getenv("STARRY_DEBUG", 0)))
if debug:
    optimize = 0
    macros = {}
    macros["STARRY_O"] = 0
    macros["STARRY_DEBUG"] = 1
else:
    optimize = 2
    macros = {}


class get_pybind_include(object):
    """
    Helper class to determine the pybind11 include path.

    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked.
    """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11

        return pybind11.get_include(self.user)


# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".cpp") as f:
        f.write("int main (int argc, char **argv) { return 0; }")
        try:
            compiler.compile([f.name], extra_postargs=["-w", flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/14/17] compiler flag.

    The newer version is prefered over c++11 (when it is available).
    """
    flags = ["-std=c++17", "-std=c++14", "-std=c++11"]

    for flag in flags:
        if has_flag(compiler, flag):
            return flag

    raise RuntimeError(
        "Unsupported compiler -- at least C++11 support is needed!"
    )


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""

    c_opts = {"msvc": ["/EHsc"], "unix": []}
    l_opts = {"msvc": [], "unix": []}

    if sys.platform == "darwin":
        darwin_opts = ["-stdlib=libc++", "-mmacosx-version-min=10.14"]
        c_opts["unix"] += darwin_opts
        l_opts["unix"] += darwin_opts

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        link_opts = self.l_opts.get(ct, [])
        if ct == "unix":
            opts.append(
                '-DVERSION_INFO="%s"' % self.distribution.get_version()
            )
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, "-fcolor-diagnostics"):
                opts.append("-fcolor-diagnostics")
        elif ct == "msvc":
            opts.append(
                '/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version()
            )
            opts.append("/Zm10")  # debug for C1060
        extra_args = ["-O%d" % optimize]
        if debug:
            extra_args += ["-g", "-Wall", "-fno-lto"]
        else:
            if has_flag(self.compiler, "-fvisibility=hidden"):
                opts.append("-fvisibility=hidden")
            extra_args += ["-g0"]
        for ext in self.extensions:
            ext.extra_compile_args = opts + extra_args
            ext.extra_link_args = link_opts + extra_args

        build_ext.build_extensions(self)


ext_modules = [
    Extension(
        "oblate._c_ops",
        ["oblate.cpp"],
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True),
            ".",
            "../../vendor/eigen_3.3.5",
            "../../vendor/boost_1_75_0",
        ],
        language="c++",
        define_macros=[(key, value) for key, value in macros.items()],
    )
]
cmdclass = {"build_ext": BuildExt}


setup(
    name="oblate",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    include_package_data=True,
    zip_safe=False,
)
