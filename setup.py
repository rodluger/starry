"""starry install script."""
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import os
import setuptools
__version__ = '0.0.2'


# Custom compiler flags
STARRY_NGRAD = 43
STARRY_MP_DIGITS = 32


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


ext_modules = [
    Extension(
        'starry',
        ['starry/pybind_interface.cpp'],
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True),
            # Path to starry headers
            "starry",
            # Path to eigen headers
            "lib/eigen_3.3.3",
            # Path to boost headers
            "lib/boost_1_66_0"
        ],
        language='c++',
        define_macros=[('STARRY_NGRAD',
                        os.getenv('STARRY_NGRAD', STARRY_NGRAD)),
                       ('STARRY_MP_DIGITS',
                        os.getenv('STARRY_MP_DIGITS', STARRY_MP_DIGITS))]
    ),
]


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
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' %
                        self.distribution.get_version())
            opts.append('-std=c++11')
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' %
                        self.distribution.get_version())
        for ext in self.extensions:
            ext.extra_compile_args = opts
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
    long_description='',
    license='GPL',
    packages=['starry'],
    ext_modules=ext_modules,
    install_requires=['matplotlib',
                      'starry_maps>=0.0.7',
                      'pybind11>=2.2'],
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
)
