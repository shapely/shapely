"""Shapely errors."""
import warnings

from shapely.lib import GEOSException, ShapelyError  # NOQA


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


class GeometryTypeError(ShapelyError, TypeError, ValueError):
    """
    An error raised when the type of the geometry in question is
    unrecognized or inappropriate.
    """

    def __init__(self, msg):
        warnings.warn(
            "GeometryTypeError will derive from ShapelyError and not TypeError or ValueError in Shapely 2.0.",
            ShapelyDeprecationWarning,
            stacklevel=2,
        )
        super().__init__(msg)
