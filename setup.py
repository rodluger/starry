"""Install script for `starry`."""
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import warnings
import setuptools
import subprocess
import sys
import os
import glob

# Are we on ReadTheDocs?
on_rtd = os.getenv("READTHEDOCS") == "True"

# Custom compiler flags
macros = dict(
    STARRY_NDIGITS=16,
    STARRY_ELLIP_MAX_ITER=200,
    STARRY_MAX_LMAX=50,
    STARRY_BCUT=1.0e-3,
    STARRY_MN_MAX_ITER=100,
    STARRY_IJ_MAX_ITER=200,
    STARRY_REFINE_J_AT=25,
    STARRY_USE_INCOMPLETE_INTEGRALS=0,
    STARRY_QUAD_POINTS=100,
    STARRY_EL2_MAX_ITER=100,
)

# Override with user values
for key, value in macros.items():
    macros[key] = os.getenv(key, value)

# Compiler optimization flag -O
optimize = int(os.getenv("STARRY_O", 2))
assert optimize in [0, 1, 2, 3], "Invalid optimization flag."
macros["STARRY_O"] = optimize

# Branching optimizations (disable on Windows)
disable_branch_optim = int(os.getenv("STARRY_BRANCHING_DISABLE_OPTIM", 0))
if sys.platform not in ["darwin", "linux"]:
    disable_branch_optim = 1
macros["STARRY_BRANCHING_DISABLE_OPTIM"] = disable_branch_optim

# Debug mode?
debug = bool(int(os.getenv("STARRY_DEBUG", 0)))
if debug:
    optimize = 0
    macros["STARRY_O"] = 0
    macros["STARRY_DEBUG"] = 1

# Numerical override at high l?
if bool(int(os.getenv("STARRY_KL_NUMERICAL", 0))):
    macros["STARRY_KL_NUMERICAL"] = 1

# Compute the Oren-Nayar (1994) expansion if the user requests it
deg = os.getenv("STARRY_OREN_NAYAR_DEG", None)
Nb = os.getenv("STARRY_OREN_NAYAR_NB", None)
if (deg is not None) or (Nb is not None):

    sys.path.insert(
        1,
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "starry",
            "_core",
            "ops",
            "lib",
            "include",
            "reflected",
        ),
    )
    import oren_nayar

    kwargs = dict()
    if deg is not None:
        kwargs["deg"] = int(deg)
    if Nb is not None:
        kwargs["Nb"] = int(Nb)
    oren_nayar.generate_header(**kwargs)


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


if int(macros["STARRY_NDIGITS"]) > 16:
    include_dirs = ["starry/_core/ops/lib/vendor/boost_1_66_0"]
else:
    include_dirs = []
if on_rtd:
    ext_modules = []
    cmdclass = {}
else:
    ext_modules = [
        Extension(
            "starry._c_ops",
            ["starry/_core/ops/lib/include/interface.cpp"],
            include_dirs=include_dirs
            + [
                # Path to pybind11 headers
                get_pybind_include(),
                get_pybind_include(user=True),
                "starry/_core/ops/lib/include",
                "starry/_core/ops/lib/vendor/eigen_3.3.5",
            ],
            language="c++",
            define_macros=[(key, value) for key, value in macros.items()],
        )
    ]
    cmdclass = {"build_ext": BuildExt}


setup(
    name="starry",
    author="Rodrigo Luger",
    author_email="rodluger@gmail.com",
    url="https://github.com/rodluger/starry",
    description="Analytic occultation light curves for astronomy.",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    license="MIT",
    packages=find_packages(),
    ext_modules=ext_modules,
    use_scm_version={
        "write_to": os.path.join("starry", "starry_version.py"),
        "write_to_template": '__version__ = "{version}"\n',
    },
    install_requires=[
        "setuptools_scm",
        "numpy>=1.19.2",
        "scipy>=1.5.0",
        "astropy>=3.1",
        "pymc3",
        "pymc3-ext",
        "matplotlib>=3.2.2",
        "ipython",
        "exoplanet>=0.4.0",
        "packaging",
    ],
    extras_require={
        "tests": [
            "parameterized",
            "nose",
            "pytest",
            "pytest-dependency",
            "pytest-env",
            "pytest-cov",
            "scikit-image",
            "pillow",
            "tqdm",
        ],
        "docs": [
            "sphinx>=1.7.5",
            "pandoc",
            "jupyter",
            "jupytext",
            "ipywidgets",
            "nbformat",
            "nbconvert",
            "rtds_action",
            "nbsphinx",
            "tqdm",
        ],
    },
    setup_requires=["setuptools_scm", "pybind11>2.4"],
    cmdclass=cmdclass,
    data_files=glob.glob("starry/img/*.png"),
    include_package_data=True,
    zip_safe=False,
)
