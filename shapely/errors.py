"""Shapely errors."""
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


class GeometryTypeError(ShapelyError):
    """
    An error raised when the type of the geometry in question is
    unrecognized or inappropriate.
    """


def __getattr__(name):
    import warnings

    if name in [
        "ReadingError",
        "WKBReadingError",
        "WKTReadingError",
        "PredicateError",
        "InvalidGeometryError",
    ]:
        warnings.warn(
            f"{name} is deprecated and will be removed in a future version. "
            "Use ShapelyError instead (functions previously raising {name} "
            "will now raise a ShapelyError instead).",
            DeprecationWarning,
            stacklevel=2,
        )
        return ShapelyError

    raise AttributeError(f"module 'shapely.errors' has no attribute '{name}'")
