"""Decorators for Shapely functions."""

import inspect
import os
import warnings
from functools import wraps

import numpy as np

from shapely import lib
from shapely.errors import UnsupportedGEOSVersionError


class requires_geos:
    """Decorator to require a minimum GEOS version."""

    def __init__(self, version):
        """Create a decorator that requires a minimum GEOS version."""
        if version.count(".") != 2:
            raise ValueError("Version must be <major>.<minor>.<patch> format")
        self.version = tuple(int(x) for x in version.split("."))

    def __call__(self, func):
        """Return the wrapped function."""
        is_compatible = lib.geos_version >= self.version
        is_doc_build = os.environ.get("SPHINX_DOC_BUILD") == "1"  # set in docs/conf.py
        if is_compatible and not is_doc_build:
            return func  # return directly, do not change the docstring

        msg = "'{}' requires at least GEOS {}.{}.{}.".format(
            func.__name__, *self.version
        )
        if is_compatible:

            @wraps(func)
            def wrapped(*args, **kwargs):
                return func(*args, **kwargs)

        else:

            @wraps(func)
            def wrapped(*args, **kwargs):
                raise UnsupportedGEOSVersionError(msg)

        doc = wrapped.__doc__
        if doc:
            # Insert the message at the first double newline
            position = doc.find("\n\n") + 2
            # Figure out the indentation level
            indent = 0
            while True:
                if doc[position + indent] == " ":
                    indent += 1
                else:
                    break
            wrapped.__doc__ = doc.replace(
                "\n\n", "\n\n{}.. note:: {}\n\n".format(" " * indent, msg), 1
            )

        return wrapped


def multithreading_enabled(func):
    """Enable multithreading.

    To do this, the writable flags of object type ndarrays are set to False.

    NB: multithreading also requires the GIL to be released, which is done in
    the C extension (ufuncs.c).
    """

    @wraps(func)
    def wrapped(*args, **kwargs):
        array_args = [
            arg for arg in args if isinstance(arg, np.ndarray) and arg.dtype == object
        ] + [
            arg
            for name, arg in kwargs.items()
            if name not in {"where", "out"}
            and isinstance(arg, np.ndarray)
            and arg.dtype == object
        ]
        old_flags = [arr.flags.writeable for arr in array_args]
        try:
            for arr in array_args:
                arr.flags.writeable = False
            return func(*args, **kwargs)
        finally:
            for arr, old_flag in zip(array_args, old_flags):
                arr.flags.writeable = old_flag

    return wrapped


def deprecate_positional(should_be_kwargs, category=DeprecationWarning):
    """Show warning if positional arguments are used that should be keyword."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # call the function first, to make sure the signature matches
            ret_value = func(*args, **kwargs)

            # check signature to see which positional args were used
            sig = inspect.signature(func)
            args_bind = sig.bind_partial(*args)
            warn_args = [
                f"`{arg}`"
                for arg in args_bind.arguments.keys()
                if arg in should_be_kwargs
            ]
            if warn_args:
                if len(warn_args) == 1:
                    plr = ""
                    isare = "is"
                    args = warn_args[0]
                else:
                    plr = "s"
                    isare = "are"
                    if len(warn_args) < 3:
                        args = " and ".join(warn_args)
                    else:
                        args = ", ".join(warn_args[:-1]) + ", and " + warn_args[-1]
                msg = (
                    f"positional argument{plr} {args} for `{func.__name__}` "
                    f"{isare} deprecated.  Please use keyword argument{plr} instead."
                )
                warnings.warn(msg, category=category, stacklevel=2)
            return ret_value

        return wrapper

    return decorator
