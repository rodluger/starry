"""starry install script."""
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import os
import setuptools
__version__ = '0.0.2'


# Custom compiler flags
macros = dict(STARRY_NGRAD=13,
              STARRY_NMULTI=32,
              STARRY_IJ_MAX_ITER=200,
              STARRY_ELLIP_MAX_ITER=200,
              STARRY_KEPLER_MAX_ITER=100)

# Override with user values
for key, value in macros.items():
    macros[key] = os.getenv(key, value)

# HACK: We should probably follow the instructions here:
# https://eigen.tuxfamily.org/dox/group__TopicStructHavingEigenMembers.html
# but it's easier to require the gradient length to be odd!
if int(macros['STARRY_NGRAD']) % 2 == 0:
    macros['STARRY_NGRAD'] = int(macros['STARRY_NGRAD']) + 1

# Enable optimization?
optimize = True
if (os.getenv('STARRY_NO_OPT', 0)):
    optimize = False


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
            "lib/boost_1_66_0",
            # Path to LBFGSpp headers
            "lib/LBFGSpp/include"
        ],
        language='c++',
        define_macros=[(key, value) for key, value in macros.items()]
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
            opts.append('-std=c++14')
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' %
                        self.distribution.get_version())
        for ext in self.extensions:
            ext.extra_compile_args = opts
            if not optimize:
                ext.extra_compile_args += ["-O0"]
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
                      'starry_maps>=0.0.12',
                      'pybind11>=2.2'],
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
)
