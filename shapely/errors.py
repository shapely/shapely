"""Shapely errors."""
from shapely.lib import GEOSException, ShapelyError, _setup_signal_checks  # NOQA
import threading


def setup_signal_checks(interval=10000, main_thread_id=None):
    """This enables Python signal checks in the ufunc inner loops.

    Doing so allows termination (using CTRL+C) of operations on large arrays of vectors.

    Parameters
    ----------
    interval : int, default 10000
        Check for interrupts every x iterations. The higher the number, the slower
        shapely will respond to a signal. However, at low values there will be a negative effect
        on performance. The default of 10000 does not have any measureable effects on performance.
    main_thread_id : int, defaults to ``threading.get_ident``
        Python signal handlers are always executed in the main Python thread of the main
        interpreter. Shapely needs to know the main thread id to prevent needless signal
        checks which would deteriorate performance in multithreading contexts.
    
    Notes
    -----
    For more information on signals consult the Python docs:

    https://docs.python.org/3/library/signal.html
    """
    if interval <= 0:
        raise ValueError("Signal checks interval must be greater than zero.")

    _setup_signal_checks(interval, main_thread_id or threading.get_ident())


class UnsupportedGEOSVersionError(ShapelyError):
    """Raised when the GEOS library version does not support a certain operation."""


class DimensionError(ShapelyError):
    """An error in the number of coordinate dimensions."""


class TopologicalError(ShapelyError):
    """A geometry is invalid or topologically incorrect."""


class ShapelyDeprecationWarning(FutureWarning):
    """
    Warning for features that will be removed or behaviour that will be
    changed in a future release.
    """


class EmptyPartError(ShapelyError):
    """An error signifying an empty part was encountered when creating a multi-part."""


class GeometryTypeError(ShapelyError):
    """
    An error raised when the type of the geometry in question is
    unrecognized or inappropriate.
    """
