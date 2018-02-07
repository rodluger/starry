#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import, \
     unicode_literals

# Version number
__version__ = "0.0.1"

# Was starry imported from setup.py?
try:
    __STARRY_SETUP__
except NameError:
    __STARRY_SETUP__ = False

if not __STARRY_SETUP__:
    # This is a regular starry run
    pass
