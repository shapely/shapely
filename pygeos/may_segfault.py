import multiprocessing
import warnings


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
            raise OSError(f"GEOS crashed with exit code {process.exitcode}.")
        else:
            return process.return_value

    return wrapper
