# -*- coding: utf-8 -*-
from inspect import getmro
import theano
import theano.tensor as tt
import numpy as np
from theano.ifelse import ifelse
from theano.tensor.extra_ops import CpuContiguous
from theano import gof
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())


__all__ = ["logger",
           "STARRY_ORTHOGRAPHIC_PROJECTION",
           "STARRY_RECTANGULAR_PROJECTION",
           "MapVector",
           "autocompile",
           "get_projection",
           "is_theano",
           "to_tensor",
           "as_contiguous_variable",
           "to_array",
           "vectorize",
           "atleast_2d",
           "cross",
           "RAxisAngle",
           "VectorRAxisAngle",
           "CheckBoundsOp",
           "RaiseValuerErrorIfOp"]


# Constants
STARRY_ORTHOGRAPHIC_PROJECTION = 0
STARRY_RECTANGULAR_PROJECTION = 1


class CompileLogMessage:
    """
    Log a brief message saying what method is currently
    being compiled and print `Done` when finished.

    """
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        logger.handlers[0].terminator = ""
        logger.info("Compiling `{0}`... ".format(self.name))

    def __exit__(self, type, value, traceback):
        logger.handlers[0].terminator = "\n"
        logger.info("Done.")


class DynamicType(object):
    """

    """
    def __call__(self, *args):
        raise NotImplementedError("This type must be subclassed.")


class MapVector(DynamicType):
    """

    """
    def __call__(self, ops):
        if ops.nw is None:
            return tt.dvector()
        else:
            return tt.dmatrix()


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
            if (not no_compile) and ((not instance.lazy) or (force_compile)):
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
                        compiled_func = theano.function(
                            [*cur_args], 
                            func(instance, *cur_args), 
                            on_unused_input='ignore'
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


def is_theano(*objs):
    """
    Return ``True`` if any of ``objs`` is a ``Theano`` object.

    """
    for obj in objs:
        for c in getmro(type(obj)):
            if c is theano.gof.graph.Node:
                return True
    return False


def to_tensor(*args):
    """
    Convert all ``args`` to tensor variables.

    """
    if len(args) == 1:
        return tt.as_tensor_variable(args[0]).astype(tt.config.floatX)
    else:
        return [tt.as_tensor_variable(arg).astype(tt.config.floatX) for arg in args]


def as_contiguous_variable(x):
    """Make `x` C-contiguous."""
    return CpuContiguous()(tt.as_tensor_variable(x))


def to_array(*args):
    """
    Convert all ``args`` to numpy arrays.

    """
    if len(args) == 1:
        return np.array(args[0], dtype=tt.config.floatX)
    else:
        return [np.array(arg, dtype=tt.config.floatX) for arg in args]


def vectorize(*args):
    """
    Vectorize all ``args`` so that they have the same length
    along the first axis.

    TODO: Add error catching if the dimensions don't agree.
    
    """
    if is_theano(*args):
        args = [arg * tt.ones(1) for arg in args]
        size = tt.max([arg.shape[0] for arg in args])
        args = [tt.repeat(arg, size // arg.shape[0], 0) for arg in args]
    else:
        args = [np.atleast_1d(arg) for arg in args]
        size = np.max([arg.shape[0] for arg in args])
        args = tuple([arg * np.ones((size,) + tuple(np.ones(len(arg.shape) - 1, dtype=int))) for arg in args])
    if len(args) == 1:
        return args[0]
    else:
        return args


def atleast_2d(arg):
    if is_theano(arg):
        return arg * tt.ones((1, 1))
    else:
        return np.atleast_2d(arg)


def cross(x, y):
    """
    Cross product of two 3-vectors.

    Based on ``https://github.com/Theano/Theano/pull/3008``
    """
    eijk = np.zeros((3, 3, 3))
    eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
    eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1
    result = tt.as_tensor_variable(
        tt.dot(tt.dot(eijk, y), x)
    )
    return result


def RAxisAngle(axis=[0, 1, 0], theta=0):
    """
    
    """
    axis = tt.as_tensor_variable(axis)
    axis /= axis.norm(2)
    cost = tt.cos(theta)
    sint = tt.sin(theta)

    return tt.reshape(tt.as_tensor_variable([
        cost + axis[0] * axis[0] * (1 - cost),
        axis[0] * axis[1] * (1 - cost) - axis[2] * sint,
        axis[0] * axis[2] * (1 - cost) + axis[1] * sint,
        axis[1] * axis[0] * (1 - cost) + axis[2] * sint,
        cost + axis[1] * axis[1] * (1 - cost),
        axis[1] * axis[2] * (1 - cost) - axis[0] * sint,
        axis[2] * axis[0] * (1 - cost) - axis[1] * sint,
        axis[2] * axis[1] * (1 - cost) + axis[0] * sint,
        cost + axis[2] * axis[2] * (1 - cost)
    ]), [3, 3])


def VectorRAxisAngle(axis=[0, 1, 0], theta=0):
    """

    """
    fn = lambda theta, axis: RAxisAngle(axis=axis, theta=theta)
    R, _ = theano.scan(fn=fn, sequences=[theta], non_sequences=[axis])
    return R


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
                "%s out of bounds: %f %s %f" % 
                (self.name, value, sign, bound)
            )


class RaiseValuerErrorIfOp(tt.Op):
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