# -*- coding: utf-8 -*-
import warnings
import aesara_theano_fallback
from aesara_theano_fallback import aesara as theano
import aesara_theano_fallback.tensor as tt
from aesara_theano_fallback import sparse as ts
from aesara_theano_fallback import change_flags, ifelse, USE_AESARA
from aesara_theano_fallback.tensor import slinalg
from aesara_theano_fallback.graph import basic, op, params_type, fg
from inspect import getmro


if USE_AESARA:
    from aesara.scan.utils import until as scan_until
else:
    try:
        from theano.scan.utils import until as scan_until
    except ModuleNotFoundError:
        from theano.scan_module.scan_utils import until as scan_until


__all__ = [
    "theano",
    "tt",
    "ts",
    "slinalg",
    "ifelse",
    "Apply",
    "COp",
    "Op",
    "Params",
    "ParamsType",
    "Node",
    "change_flags",
    "floatX",
    "evaluator",
    "scan_until",
    "USE_AESARA",
]

# Suppress third-party deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pymc3")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="theano")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="aesara")

# Set double precision
floatX = "float64"

# Compatibility imports
Node = basic.Node
Apply = basic.Apply
Op = op.Op
COp = op.ExternalCOp
Params = params_type.Params
ParamsType = params_type.ParamsType
MissingInputError = fg.MissingInputError
theano.config.floatX = floatX

# This helps prevent defaulting to float32
theano.config.cast_policy = "numpy+floatX"


def is_tensor(*objs):
    """Return ``True`` if any of ``objs`` is a ``Theano`` object."""
    for obj in objs:
        for c in getmro(type(obj)):
            if c is Node:
                return True
    return False


def evaluator(**kwargs):
    """
    Return a function to evaluate theano tensors.

    Works inside a `pymc3` model if a `point` is provided.
    Lazily imports `pymc3` to minimize overhead.

    """
    # Store the kwargs
    kwargs_point = kwargs.get("point", None)
    kwargs_model = kwargs.get("model", None)

    if kwargs_point is not None:

        # User provided a point

        import pymc3 as pm
        import pymc3_ext as pmx

        point = kwargs_point
        model = kwargs_model
        if model is None:
            model = pm.Model.get_context()

        def get_val(x):
            if is_tensor(x):
                return pmx.eval_in_model(x, model=model, point=point)
            else:
                return x

    else:

        # No point provided

        def get_val(x):

            if is_tensor(x):

                try:

                    # Try to directly evaluate it

                    return x.eval()

                except MissingInputError as e:

                    # That didn't work. Perhaps we are in a pymc3 model
                    # context, but the user didn't provide a point?

                    import pymc3 as pm
                    import pymc3_ext as pmx

                    try:
                        model = kwargs_model
                        if model is None:
                            model = pm.Model.get_context()
                    except TypeError:
                        raise ValueError(
                            "Missing input for variable {}, and no pymc3 model found.".format(
                                x
                            )
                        )

                    # Warn the user that we're using the test point
                    warnings.warn(
                        "Detected pymc3 model context, but no point provided. "
                        "Evaluating at test_point."
                    )

                    return pmx.eval_in_model(
                        x, model=model, point=model.test_point
                    )

            else:

                return x

    return get_val
