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
    from theano.graph.fg import MissingInputError
except ImportError:
    from theano.gof.graph import Apply, Node
    from theano.gof.op import COp, Op
    from theano.gof.params_type import Params, ParamsType
    from theano.gof.fg import MissingInputError
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
    # Store the kwargs
    kwargs_point = kwargs.pop("point", None)
    kwargs_model = kwargs.pop("model", None)

    if kwargs_point is not None:

        # User provided a point

        import pymc3 as pm
        import pymc3_ext as pmx

        point = kwargs_point
        model = kwargs_model
        if model is None:
            model = pm.Model.get_context()
        get_val = lambda x: pmx.eval_in_model(x, model=model, point=point)

    else:

        # No point provided

        def get_val(x):

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
                    "Detected pymc3 model context, but no point provided. Evaluating at test_point."
                )

                return pmx.eval_in_model(
                    x, model=model, point=model.test_point
                )

    return get_val
