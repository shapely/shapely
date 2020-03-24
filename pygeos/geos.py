from . import lib
from functools import wraps


class UnsupportedGEOSOperation(ImportError):
    pass


class requires_geos:
    def __init__(self, version):
        if version.count(".") != 2:
            raise ValueError("Version must be <major>.<minor>.<patch> format")
        self.version = tuple(int(x) for x in version.split("."))

    def __call__(self, func):
        if lib.geos_version < self.version:
            msg = "'{}' requires at least GEOS {}.{}.{}".format(func.__name__, *self.version)

            @wraps(func)
            def wrapped(*args, **kwargs):
                raise UnsupportedGEOSOperation(msg)

            wrapped.__doc__ = msg
            return wrapped
        else:
            return func
