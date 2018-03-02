#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import os
from setuptools import setup, Extension

# Hackishly inject a constant into builtins to enable importing of the
# module in "setup" mode. Stolen from `kplr`
import sys
if sys.version_info[0] < 3:
    import __builtin__ as builtins
else:
    import builtins
builtins.__STARRY_SETUP__ = True
import starry
from starry.build import build_ext  # NOQA

ext = Extension("starry.interface",
                sources=[os.path.join("starry", "interface.cpp")],
                language="c++")

long_description = \
    """Analytic occultation light curves for astronomy."""

# Setup!
setup(name='starry',
      version=starry.__version__,
      description='Analytic occultation light curves for astronomy.',
      long_description=long_description,
      classifiers=[
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: GNU General Public License (GPL)',
          'Programming Language :: Python',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering :: Astronomy',
      ],
      url='http://github.com/rodluger/starry',
      author='Rodrigo Luger',
      author_email='rodluger@uw.edu',
      license='GPL',
      packages=['starry'],
      ext_modules=[ext],
      install_requires=[
                        'numpy',
                        'scipy',
                        'matplotlib',
                        'tqdm',
                        'sympy',
                        'mpmath',
                        'healpy',
                        'pybind11'],
      dependency_links=[],
      scripts=[],
      include_package_data=True,
      cmdclass=dict(build_ext=build_ext),
      zip_safe=False
      )
