"""starry2 install script."""
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import os
import glob
import setuptools
__version__ = '1.0.0'

# Module bits
_STARRY_DEFAULT_DOUBLE_ =  1
_STARRY_DEFAULT_MULTI_ =   2
_STARRY_SPECTRAL_DOUBLE_ = 4
_STARRY_SPECTRAL_MULTI_ =  8
_STARRY_TEMPORAL_DOUBLE_ = 16
_STARRY_TEMPORAL_MULTI_ =  32

# Custom compiler flags
macros = dict(STARRY_NMULTI=32,
              STARRY_ELLIP_MAX_ITER=200,
              STARRY_MAX_LMAX=50,
              STARRY_BCUT=1.e-3,
              STARRY_MN_MAX_ITER=100)

# Override with user values
for key, value in macros.items():
    macros[key] = os.getenv(key, value)

# Don't compute dF/du?
no_dfdu = int(os.getenv('STARRY_KEEP_DFDU_AS_DFDG', 0))
if no_dfdu:
    macros["STARRY_KEEP_DFDU_AS_DFDG"] = 1

# Compiler optimization flag -O
optimize = int(os.getenv('STARRY_O', 2))
assert optimize in [0, 1, 2, 3], "Invalid optimization flag."
macros["STARRY_O"] = optimize

# Debug mode?
debug = bool(int(os.getenv('STARRY_DEBUG', 0)))
if debug:
    optimize = 0
    macros["STARRY_O"] = 0
    macros["STARRY_DEBUG"] = 1

# Module bitsum (1 + 2 + 4 + 8 + 16 + 32 = 63)
bitsum = int(os.getenv('STARRY_BITSUM', 63))

class get_pybind_include(object):
    """
    Helper class to determine the pybind11 include path.

    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked.
    """

    def __init__(self, user=False):
        """Init."""
        self.user = user

    def __str__(self):
        """Str."""
        import pybind11
        return pybind11.get_include(self.user)


def get_ext(module='starry2._starry_default_double', name='_STARRY_DEFAULT_DOUBLE_'):
    include_dirs = [
        get_pybind_include(),
        get_pybind_include(user=True),
        "include",
        "lib/eigen_3.3.5",
        # "lib/LBFGSpp/include" TODO
    ]
    if 'MULTI' in name:
        include_dirs += ["lib/boost_1_66_0"]
    return Extension(
        module,
        ['include/pybind/interface.cpp'],
        include_dirs=include_dirs,
        language='c++',
        define_macros=[(name, 1)] +
                      [(key, value) for key, value in macros.items()]
    )

# Figure out which modules to compile (default all)
ext_modules = []
if (bitsum & _STARRY_DEFAULT_DOUBLE_):
    ext_modules.append(get_ext('starry2._starry_default_double', '_STARRY_DEFAULT_DOUBLE_'))
if (bitsum & _STARRY_DEFAULT_MULTI_):
    ext_modules.append(get_ext('starry2._starry_default_multi', '_STARRY_DEFAULT_MULTI_'))
if (bitsum & _STARRY_SPECTRAL_DOUBLE_):
    ext_modules.append(get_ext('starry2._starry_spectral_double', '_STARRY_SPECTRAL_DOUBLE_'))
if (bitsum & _STARRY_SPECTRAL_MULTI_):
    ext_modules.append(get_ext('starry2._starry_spectral_multi', '_STARRY_SPECTRAL_MULTI_'))
if (bitsum & _STARRY_TEMPORAL_DOUBLE_):
    ext_modules.append(get_ext('starry2._starry_temporal_double', '_STARRY_TEMPORAL_DOUBLE_'))
if (bitsum & _STARRY_TEMPORAL_MULTI_):
    ext_modules.append(get_ext('starry2._starry_temporal_multi', '_STARRY_TEMPORAL_MULTI_'))

# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """
    Check if flag name is supported.

    Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""

    c_opts = {
        'msvc': ['/EHsc'],
        'unix': [],
    }

    if sys.platform == 'darwin':
        c_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.7']

    def build_extensions(self):
        """Build the extensions."""
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        if not any(f.startswith("-std=") for f in opts):
            if has_flag(self.compiler, "-std=c++14"):
                opts.append('-std=c++14')
            elif has_flag(self.compiler, "-std=c++11"):
                opts.append('-std=c++11')
            else:
                raise RuntimeError("C++11 or 14 is required to compile starry")
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' %
                        self.distribution.get_version())
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' %
                        self.distribution.get_version())
        for ext in self.extensions:
            ext.extra_compile_args = list(opts + ext.extra_compile_args)
            ext.extra_compile_args += ["-O%d" % optimize]
            ext.extra_compile_args += ["-Wextra",
                                       "-Wno-unused-parameter",
                                       "-Wno-unused-lambda-capture",
                                       "-Wpedantic"]
            if debug:
                ext.extra_compile_args += ["-g"]
            else:
                ext.extra_compile_args += ["-g0"]
            if sys.platform == "darwin":
                ext.extra_compile_args += ["-march=native",
                                           "-mmacosx-version-min=10.9"]
                ext.extra_link_args += ["-march=native",
                                        "-mmacosx-version-min=10.9"]
        build_ext.build_extensions(self)


setup(
    name='starry2',
    version=__version__,
    author='Rodrigo Luger',
    author_email='rodluger@gmail.com',
    url='https://github.com/rodluger/starry',
    description='Analytic occultation light curves for astronomy.',
    license='GPL',
    packages=['starry2'],
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.2'],
    cmdclass={'build_ext': BuildExt},
    data_files=[], #glob.glob('starry/maps/*.jpg')],
    include_package_data=True,
    zip_safe=False,
)
