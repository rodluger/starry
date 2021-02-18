# -*- coding: utf-8 -*-

__all__ = [
    "Apply",
    "COp",
    "Op",
    "Params",
    "ParamsType",
    "Node",
    "change_flags",
]

# Suppress third-party deprecation warnings
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning, module="pymc3")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="theano")

# Import theano interface
try:
    from theano.graph.basic import Apply, Node
    from theano.graph.op import ExternalCOp as COp
    from theano.graph.op import Op
    from theano.graph.params_type import Params, ParamsType
except ImportError:
    from theano.gof.graph import Apply, Node
    from theano.gof.op import COp, Op
    from theano.gof.params_type import Params, ParamsType
try:
    import theano

    change_flags = theano.config.change_flags
except:
    from theano.configparser import change_flags

# Force double precision
import theano.tensor as tt

tt.config.floatX = "float64"
