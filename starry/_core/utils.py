# -*- coding: utf-8 -*-
from .. import config
import theano
import theano.tensor as tt
import numpy as np
from theano.configparser import change_flags
from inspect import getmro
from functools import wraps
import logging

logger = logging.getLogger("starry.ops")

__all__ = ["logger", "autocompile"]


integers = (int, np.int, np.int16, np.int32, np.int64)


def is_theano(*objs):
    """Return ``True`` if any of ``objs`` is a ``Theano`` object."""
    for obj in objs:
        for c in getmro(type(obj)):
            if c is theano.gof.graph.Node:
                return True
    return False


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


def _get_type(arg):
    """
    Get the theano tensor type corresponding to `arg`.

    Note that arg must be one of the following:
        - a theano tensor
        - an integer (`int`, `np.int`, `np.int16`, `np.int32`, `np.int64`)
        - a numpy boolean (`np.array(True)`, `np.array(False)`)
        - a numpy float array with ndim equal to 0, 1, 2, or 3

    # TODO: Cast lists to arrays and floats to np.array(float)

    """
    ttype = type(arg)
    if is_theano(arg):
        return ttype
    else:
        if ttype in integers:
            return tt.iscalar
        elif hasattr(arg, "ndim"):
            if arg.ndim == 0:
                if arg.dtype is np.array(True).dtype:
                    return tt.bscalar
                else:
                    return tt.dscalar
            elif arg.ndim == 1:
                return tt.dvector
            elif arg.ndim == 2:
                return tt.dmatrix
            elif arg.ndim == 3:
                return tt.dtensor3
            else:
                raise NotImplementedError(
                    "Invalid array dimension passed to @autocompile: {}.".format(
                        arg.ndim
                    )
                )
        else:
            raise NotImplementedError(
                "Invalid argument type passed to @autocompile: {}.".format(
                    ttype
                )
            )


def autocompile(func):
    """
    Wrap the method `func` and return a compiled version
    if none of the arguments are tensors.

    """

    @wraps(func)  # inherit docstring
    def wrapper(instance, *args):

        if is_theano(*args):

            # Just return the function as is
            return func(instance, *args)

        else:

            # Determine the argument types
            arg_types = tuple([_get_type(arg) for arg in args])

            # Get a unique name for the compiled function
            cname = "{}_{}".format(func.__name__, hex(hash(arg_types)))

            # Compile the function if needed & cache it
            if not hasattr(instance, cname):

                dummy_args = [arg_type() for arg_type in arg_types]

                # Compile the function
                with CompileLogMessage(func.__name__):
                    with change_flags(compute_test_value="off"):
                        compiled_func = theano.function(
                            [*dummy_args],
                            func(instance, *dummy_args),
                            on_unused_input="ignore",
                            profile=config.profile,
                        )
                    setattr(instance, cname, compiled_func)

            # Return the compiled version
            return getattr(instance, cname)(*args)

    return wrapper
