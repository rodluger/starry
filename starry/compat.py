# -*- coding: utf-8 -*-

__all__ = [
    "Apply",
    "COp",
    "Op",
    "Params",
    "ParamsType",
    "Node",
    "change_flags",
    "evaluator",
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


def evaluator(**kwargs):
    """
    Return a function to evaluate theano tensors.

    Works inside a `pymc3` model if a `point` is provided.
    Lazily imports `pymc3` to minimize overhead.

    """
    if kwargs.get("point", None) is not None:

        import pymc3 as pm
        import pymc3_ext as pmx

        point = kwargs.get("point")
        model = kwargs.get("model", pm.Model.get_context())
        get_val = lambda x: pmx.eval_in_model(x, model=model, point=point)

    else:

        get_val = lambda x: x.eval()

    return get_val
