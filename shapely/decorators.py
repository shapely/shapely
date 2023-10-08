import multiprocessing
import os
import warnings
from functools import wraps

import numpy as np

from shapely import lib
from shapely.errors import UnsupportedGEOSVersionError


class requires_geos:
    def __init__(self, version):
        if version.count(".") != 2:
            raise ValueError("Version must be <major>.<minor>.<patch> format")
        self.version = tuple(int(x) for x in version.split("."))

    def __call__(self, func):
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
            indent = 2
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


class ReturningProcess(multiprocessing.Process):
    """A Process with an added Pipe for getting the return_value or exception."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pconn, self._cconn = multiprocessing.Pipe()
        self._result = {}

    def run(self):
        if not self._target:
            return
        try:
            with warnings.catch_warnings(record=True) as w:
                return_value = self._target(*self._args, **self._kwargs)
            self._cconn.send({"return_value": return_value, "warnings": w})
        except Exception as e:
            self._cconn.send({"exception": e, "warnings": w})

    @property
    def result(self):
        if not self._result and self._pconn.poll():
            self._result = self._pconn.recv()
        return self._result

    @property
    def exception(self):
        return self.result.get("exception")

    @property
    def warnings(self):
        return self.result.get("warnings", [])

    @property
    def return_value(self):
        return self.result.get("return_value")


def may_segfault(func):
    """The wrapped function will be called in another process.

    If the execution crashes with a segfault or sigabort, an OSError
    will be raised.

    Note: do not use this to decorate a function at module level, because this
    will render the function un-Picklable so that multiprocessing fails on OSX/Windows.

    Instead, use it like this:

    >>> def some_unstable_func():
    ...     ...
    >>> some_func = may_segfault(some_unstable_func)
    """

    def wrapper(*args, **kwargs):
        process = ReturningProcess(target=func, args=args, kwargs=kwargs)
        process.start()
        process.join()
        for w in process.warnings:
            warnings.warn_explicit(
                w.message,
                w.category,
                w.filename,
                w.lineno,
            )
        if process.exception:
            raise process.exception
        elif process.exitcode != 0:
            raise OSError(f"Function crashed with exit code {process.exitcode}.")
        else:
            return process.return_value

    return wrapper
