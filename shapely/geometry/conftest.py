"""Autouse fixtures for doctests."""

import pytest

from shapely.geometry.linestring import LineString


@pytest.fixture(autouse=True)
def add_linestring(doctest_namespace):
    """Add LineString to the doctest namespace."""
    doctest_namespace["LineString"] = LineString
