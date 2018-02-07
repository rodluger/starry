#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
from setuptools import setup

# Hackishly inject a constant into builtins to enable importing of the
# module in "setup" mode. Stolen from `kplr`
import sys
if sys.version_info[0] < 3:
    import __builtin__ as builtins
else:
    import builtins
builtins.__STARRY_SETUP__ = True
import starry

long_description = \
    """STARRY."""

# Setup!
setup(name='starry',
      version=starry.__version__,
      description='STARRY',
      long_description=long_description,
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering :: Astronomy',
      ],
      url='http://github.com/rodluger/starry',
      author='Rodrigo Luger',
      author_email='rodluger@uw.edu',
      license='GPL',
      packages=['starry'],
      install_requires=[],
      dependency_links=[],
      scripts=[],
      include_package_data=True,
      zip_safe=False,
      test_suite='nose.collector',
      tests_require=['nose']
      )
