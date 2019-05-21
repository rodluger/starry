"""Install script for `starry`."""
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import os
import glob
import setuptools

# Figure out the current version
if sys.version_info[0] < 3:
    import __builtin__ as builtins
else:
    import builtins
builtins.__STARRY_SETUP__ = True
from starry import __version__

# Custom compiler flags
macros = dict(
    STARRY_NDIGITS=16,
    STARRY_ELLIP_MAX_ITER=200,
    STARRY_MAX_LMAX=50,
    STARRY_BCUT=1.e-3,
    STARRY_MN_MAX_ITER=100,
    STARRY_IJ_MAX_ITER=200,
    STARRY_REFINE_J_AT=25
)

# Override with user values
for key, value in macros.items():
    macros[key] = os.getenv(key, value)

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


# Get C extension
def get_ext():
    include_dirs = [
        get_pybind_include(),
        get_pybind_include(user=True),
        "starry/include",
        "lib/eigen_3.3.5"
    ]
    if macros["STARRY_NDIGITS"] > 16:
        include_dirs += ["lib/boost_1_66_0"]
    return Extension(
        'starry._c_ops',
        ['starry/include/interface.cpp'],
        include_dirs=include_dirs,
        language='c++',
        define_macros=[(key, value) for key, value in macros.items()]
    )
ext_modules = [get_ext()]


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
                                       "-Wpedantic",
                                       "-Wno-unused-parameter",  # DEBUG disable this
                                       "-Wno-unused-lambda-capture",
                                       "-Wno-unused-local-typedef"]
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
    name='starry',
    version=__version__,
    author='Rodrigo Luger',
    author_email='rodluger@gmail.com',
    url='https://github.com/rodluger/starry',
    description='Analytic occultation light curves for astronomy.',
    license='GPL',
    packages=['starry'],
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.2',
                      'theano>=1.0.4',
                      'pillow',
                      'ipython'],
    cmdclass={'build_ext': BuildExt},
    data_files=glob.glob('starry/img/*.jpg'),
    include_package_data=True,
    zip_safe=False,
    extras_require={
        'healpy':  ['healpy>=1.12.8'],
        'exoplanet': ['exoplanet>=0.1.4'],
        'pymc3': ['pymc3>=3.6']
    }
)