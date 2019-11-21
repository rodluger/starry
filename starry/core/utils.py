# -*- coding: utf-8 -*-
from .. import config
import theano
import theano.tensor as tt
import numpy as np
from theano.configparser import change_flags
import logging

logger = logging.getLogger("starry.ops")

__all__ = ["logger", "DynamicType", "autocompile"]


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
