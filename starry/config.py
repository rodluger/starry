# -*- coding: utf-8 -*-
import theano.tensor as tt

"""Floating-point precision for operations on the Python side."""
tt.config.floatX = "float64"


"""Indicates whether or not the map evaluates things lazily.

If True, all attributes and method return values are unevaluated 
``theano`` nodes. This is particularly useful for model building and 
integration with ``pymc3``. In lazy mode, call the ``.eval()`` method 
on any ``theano`` node to compute and return its numerical value. 

If False, ``starry`` will automatically compile methods called by the 
user, and all methods will return numerical values as in the previous 
version of the code.
"""
lazy = True
