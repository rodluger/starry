# -*- coding: utf-8 -*-
import warnings

import numpy.distutils
from inspect import getmro


# Monkeypatch for
#   AttributeError: module 'numpy.distutils.__config__' has no attribute 'blas_opt_info'
# See
#   https://stackoverflow.com/a/72326831
try:
    numpy.distutils.__config__.blas_opt_info = (
        numpy.distutils.__config__.blas_ilp64_opt_info
    )
except Exception:
    pass


import pytensor
# TODO: Revert to tt and ts? pt and ps mark transition in a cleaner manner and more furture-proof
import pytensor.tensor as pt
from pytensor import sparse as ps
from pytensor import change_flags, ifelse
from pytensor.tensor import slinalg
from pytensor.graph import basic, op, fg
from pytensor.link.c import params_type
from pytensor.scan.utils import until as scan_until


__all__ = [
    "pytensor",
    "pt",
    "ps",
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
]

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
pytensor.config.floatX = floatX

# This helps prevent defaulting to float32
pytensor.config.cast_policy = "numpy+floatX"


def is_tensor(*objs):
    """Return ``True`` if any of ``objs`` is a ``pytensor`` object."""
    for obj in objs:
        for c in getmro(type(obj)):
            if c is Node:
                return True
    return False


def evaluator(**kwargs):
    """
    Return a function to evaluate pytensor tensors.

    Works inside a `pymc` model if a `point` is provided.
    Lazily imports `pymc` to minimize overhead.

    """
    # Store the kwargs
    kwargs_point = kwargs.get("point", None)
    kwargs_model = kwargs.get("model", None)

    if kwargs_point is not None:

        # User provided a point

        import pymc as pm
        import pymc_ext as pmx

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

                    import pymc as pm
                    import pymc_ext as pmx

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
                        "Detected pymc model context, but no point provided. "
                        "Evaluating at test_point."
                    )

                    return pmx.eval_in_model(
                        x, model=model, point=model.test_point
                    )

            else:

                return x

    return get_val
