"""Shapely errors."""


class ShapelyError(Exception):
    """Base error class."""


class UnsupportedGEOSVersionError(ShapelyError):
    """Raised when the system's GEOS library version is unsupported."""


class ReadingError(ShapelyError):
    """A WKT or WKB reading error."""


class WKBReadingError(ReadingError):
    """A WKB reading error."""


class WKTReadingError(ReadingError):
    """A WKT reading error."""


class DimensionError(ShapelyError):
    """An error in the number of coordinate dimensions."""


class TopologicalError(ShapelyError):
    """A geometry is invalid or topologically incorrect."""


class PredicateError(ShapelyError):
    """A geometric predicate has failed to return True/False."""
