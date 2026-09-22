"""Tests of Shapely's exceptions."""

import pytest

from shapely import LineString
from shapely.errors import GEOSException
from shapely.lib import _ShapelyValueError


def test_value_error():
    """_ShapelyValueError is backward compatible."""
    # New usage.
    with pytest.raises(ValueError):
        raise _ShapelyValueError("invalid arg")

    # Older usage.
    with pytest.raises(GEOSException):
        raise _ShapelyValueError("invalid arg")


def test_geos_finish():
    """_ShapelyValueError is used by GEOS_FINISH macro."""
    # New usage.
    with pytest.raises(ValueError):
        LineString([(1, 2)])

    # Older usage.
    with pytest.raises(GEOSException):
        LineString([(1, 2)])
