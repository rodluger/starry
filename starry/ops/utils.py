# -*- coding: utf-8 -*-
from .. import config
from ..constants import *
import theano
import theano.tensor as tt
import numpy as np
from theano.configparser import change_flags
from theano import gof
from scipy.linalg import block_diag as scipy_block_diag
import theano.tensor.slinalg as sla
import logging
import scipy

logger = logging.getLogger("starry.ops")

__all__ = [
    "logger",
    "DynamicType",
    "autocompile",
    "get_projection",
    "RAxisAngle",
    "CheckBoundsOp",
    "RaiseValueErrorOp",
    "RaiseValueErrorIfOp",
    "math",
]


class CompileLogMessage:
    """
    Log a brief message saying what method is currently
    being compiled and print `Done` when finished.

    """

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        config.rootHandler.terminator = ""
        logger.info("Compiling `{0}`... ".format(self.name))

    def __exit__(self, type, value, traceback):
        config.rootHandler.terminator = "\n"
        logger.info("Done.")


class DynamicType(object):
    """

    """

    def __init__(self, code):
        self._code = code

    def __call__(self, instance):
        return eval(self._code)


class autocompile(object):
    """
    A decorator used to automatically compile methods into Theano functions
    if the user disables lazy evaluation.

    """

    def __init__(self, name, *args):
        """
        Initialize the decorator.

        Arguments:
            name (str): The name of the decorated method
            *args (tuple): Theano types corresponding to each of the
                arguments of the method.
        """
        self.args = list(args)
        self.name = name
        self.compiled_name = "_compiled_{0}".format(name)

    def __call__(self, func):
        """
        Wrap the method `func` and return a compiled version if `lazy==False`.

        """

        def wrapper(instance, *args, force_compile=False, no_compile=False):
            """
            The magic happens in here.

            """
            if (not no_compile) and ((not config.lazy) or (force_compile)):
                # Compile the function if needed & cache it
                if not hasattr(instance, self.compiled_name):

                    cur_args = list(self.args)
                    # Evaluate any dynamic types. These are tensors
                    # whose types depend on specific properties of the
                    # `Ops` instance that are evaluated at run time.
                    for i, arg in enumerate(cur_args):
                        if isinstance(arg, DynamicType):
                            cur_args[i] = arg(instance)

                    with CompileLogMessage(self.name):
                        with change_flags(compute_test_value="off"):
                            compiled_func = theano.function(
                                [*cur_args],
                                func(instance, *cur_args),
                                on_unused_input="ignore",
                            )
                        setattr(instance, self.compiled_name, compiled_func)

                # Return the compiled version
                return getattr(instance, self.compiled_name)(*args)
            else:
                # Just return the function as is
                return func(instance, *args)

        # Store the function info
        wrapper.args = self.args
        wrapper.func = func
        return wrapper


def get_projection(projection):
    """

    """
    if projection.lower().startswith("rect"):
        projection = STARRY_RECTANGULAR_PROJECTION
    elif projection.lower().startswith("ortho"):
        projection = STARRY_ORTHOGRAPHIC_PROJECTION
    else:
        raise ValueError("Unknown map projection.")
    return projection


def _RAxisAngle(axis=[0, 1, 0], theta=0):
    """

    """
    axis = tt.as_tensor_variable(axis)
    axis /= axis.norm(2)
    cost = tt.cos(theta)
    sint = tt.sin(theta)

    return tt.reshape(
        tt.as_tensor_variable(
            [
                cost + axis[0] * axis[0] * (1 - cost),
                axis[0] * axis[1] * (1 - cost) - axis[2] * sint,
                axis[0] * axis[2] * (1 - cost) + axis[1] * sint,
                axis[1] * axis[0] * (1 - cost) + axis[2] * sint,
                cost + axis[1] * axis[1] * (1 - cost),
                axis[1] * axis[2] * (1 - cost) - axis[0] * sint,
                axis[2] * axis[0] * (1 - cost) - axis[1] * sint,
                axis[2] * axis[1] * (1 - cost) + axis[0] * sint,
                cost + axis[2] * axis[2] * (1 - cost),
            ]
        ),
        [3, 3],
    )


def RAxisAngle(axis=[0, 1, 0], theta=0):
    """

    """
    if hasattr(theta, "ndim") and theta.ndim > 0:
        fn = lambda theta, axis: _RAxisAngle(axis=axis, theta=theta)
        R, _ = theano.scan(fn=fn, sequences=[theta], non_sequences=[axis])
        return R
    else:
        return _RAxisAngle(axis=axis, theta=theta)


class CheckBoundsOp(tt.Op):
    """

    """

    def __init__(self, lower=-np.inf, upper=np.inf, name=None):
        self.lower = lower
        self.upper = upper
        if name is None:
            self.name = "parameter"
        else:
            self.name = name

    def make_node(self, *inputs):
        inputs = [tt.as_tensor_variable(inputs[0])]
        outputs = [inputs[0].type()]
        return gof.Apply(self, inputs, outputs)

    def infer_shape(self, node, shapes):
        return [shapes[0]]

    def perform(self, node, inputs, outputs):
        outputs[0][0] = inputs[0]
        if np.any((inputs[0] < self.lower) | (inputs[0] > self.upper)):
            low = np.where((inputs[0] < self.lower))[0]
            high = np.where((inputs[0] > self.upper))[0]
            if len(low):
                value = inputs[0][low[0]]
                sign = "<"
                bound = self.lower
            else:
                value = inputs[0][high[0]]
                sign = ">"
                bound = self.upper
            raise ValueError(
                "%s out of bounds: %f %s %f" % (self.name, value, sign, bound)
            )


def RaiseValueErrorOp(msg, shape):
    return tt.as_tensor_variable(np.empty(shape)) * tt.RaiseValueErrorIfOp(
        msg
    )(True)


class RaiseValueErrorIfOp(tt.Op):
    """

    """

    def __init__(self, message=None):
        self.message = message

    def make_node(self, *inputs):
        condition = inputs
        inputs = [tt.as_tensor_variable(condition)]
        outputs = [tt.TensorType(tt.config.floatX, ())()]
        return gof.Apply(self, inputs, outputs)

    def infer_shape(self, node, shapes):
        return [()]

    def perform(self, node, inputs, outputs):
        outputs[0][0] = np.array(0.0)
        if inputs[0]:
            raise ValueError(self.message)

    def grad(self, inputs, gradients):
        # TODO: Is this actually necessary?
        return [inputs[0] * 0.0]


class MathType(type):
    """Wrapper for theano/numpy functions."""

    def cholesky(cls, *args, **kwargs):
        if config.lazy:
            return sla.cholesky(*args, **kwargs)
        else:
            return scipy.linalg.cholesky(*args, **kwargs, lower=True)

    def atleast_2d(cls, arg):
        if config.lazy:
            return arg * tt.ones((1, 1))
        else:
            return np.atleast_2d(arg)

    def vectorize(cls, *args):
        """
        Vectorize all ``args`` so that they have the same length
        along the first axis.

        TODO: Add error catching if the dimensions don't agree.

        """
        if config.lazy:
            args = [arg * tt.ones(1) for arg in args]
            size = tt.max([arg.shape[0] for arg in args])
            args = [tt.repeat(arg, size // arg.shape[0], 0) for arg in args]
        else:
            args = [np.atleast_1d(arg) for arg in args]
            size = np.max([arg.shape[0] for arg in args])
            args = tuple(
                [
                    arg
                    * np.ones(
                        (size,) + tuple(np.ones(len(arg.shape) - 1, dtype=int))
                    )
                    for arg in args
                ]
            )
        if len(args) == 1:
            return args[0]
        else:
            return args

    def cross(x, y):
        """Cross product of two 3-vectors.

        Based on ``https://github.com/Theano/Theano/pull/3008``
        """
        if config.lazy:
            eijk = np.zeros((3, 3, 3))
            eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
            eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1
            return tt.as_tensor_variable(tt.dot(tt.dot(eijk, y), x))
        else:
            return np.cross(x, y)

    def cast(cls, *args):
        if config.lazy:
            return cls.to_tensor(*args)
        else:
            if len(args) == 1:
                return np.array(args[0], dtype=tt.config.floatX)
            else:
                return [np.array(arg, dtype=tt.config.floatX) for arg in args]

    def to_array_or_tensor(cls, x):
        if config.lazy:
            return tt.as_tensor_variable(x)
        else:
            return np.array(x)

    def block_diag(self, *mats):
        if config.lazy:
            N = [mat.shape[0] for mat in mats]
            Nsum = tt.sum(N)
            res = tt.zeros((Nsum, Nsum), dtype=theano.config.floatX)
            n = 0
            for mat in mats:
                inds = slice(n, n + mat.shape[0])
                res = tt.set_subtensor(res[tuple((inds, inds))], mat)
                n += mat.shape[0]
            return res
        else:
            return scipy_block_diag(*mats)

    def to_tensor(cls, *args):
        """Convert all ``args`` to Theano tensor variables.

        Converts to tensor regardless of whether `config.lazy` is True or False.
        """
        if len(args) == 1:
            return tt.as_tensor_variable(args[0]).astype(tt.config.floatX)
        else:
            return [
                tt.as_tensor_variable(arg).astype(tt.config.floatX)
                for arg in args
            ]

    def __getattr__(cls, attr):
        if config.lazy:
            return getattr(tt, attr)
        else:
            return getattr(np, attr)


class math(metaclass=MathType):
    """Alias for ``numpy`` or ``theano.tensor``."""

    pass
