from contextlib import contextmanager
from functools import wraps
import numpy as np
from . import lib


class UnsupportedGEOSOperation(ImportError):
    pass


class requires_geos:
    def __init__(self, version):
        if version.count(".") != 2:
            raise ValueError("Version must be <major>.<minor>.<patch> format")
        self.version = tuple(int(x) for x in version.split("."))

    def __call__(self, func):
        if lib.geos_version < self.version:
            msg = "'{}' requires at least GEOS {}.{}.{}".format(
                func.__name__, *self.version
            )

            @wraps(func)
            def wrapped(*args, **kwargs):
                raise UnsupportedGEOSOperation(msg)

            wrapped.__doc__ = msg
            return wrapped
        else:
            return func


def multithreading_enabled(func):
    """Prepare multithreading by setting the writable flags of object type
    ndarrays to False.

    NB: multithreading also requires the GIL to be released, which is done in
    the C extension (ufuncs.c)."""

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
