"""svgtree module tests."""

from xml.etree import ElementTree as ET

import pytest

from shapely import Point, LinearRing, LineString, Polygon, set_precision
from shapely.geos import geos_version
from shapely.svgtree import SVGTree


def test_point():
    elem = SVGTree.fromgeom(Point(0, 0))
    assert (
        ET.tostring(elem, encoding="unicode")
        == '<g><circle cx="0.0" cy="0.0" r="1.0" /></g>'
    )


def test_linestring():
    elem = SVGTree.fromgeom(LineString([(0, 0), (1, 1)]))
    assert (
        ET.tostring(elem, encoding="unicode")
        == '<g><path d="M 0.0,0.0 L 1.0,1.0" /></g>'
    )


def test_linearring():
    elem = SVGTree.fromgeom(LinearRing([(0, 0), (100, 100), (100, 0)]))
    assert (
        ET.tostring(elem, encoding="unicode")
        == '<g><path d="M 0.0,0.0 L 100.0,100.0 L 100.0,0.0 Z" /></g>'
    )


@pytest.mark.skipif(
    geos_version < (3, 9, 0),
    reason="GEOS >= 3.9.0 is required to test order of interior ring coordinates.",
)
def test_polygon():
    elem = SVGTree.fromgeom(
        set_precision(
            Polygon(
                [(0, 0), (100, 100), (100, 0)],
                [Point(75, 25).buffer(20, quadsegs=2).exterior],
            ),
            0.01,
        )
    )
    assert (
        ET.tostring(elem, encoding="unicode")
        == '<g><path d="M 100.0,100.0 L 100.0,0.0 L 0.0,0.0 Z  M 89.14,39.14 L 75.0,45.0 L 60.86,39.14 L 55.0,25.0 L 60.86,10.86 L 75.0,5.0 L 89.14,10.86 L 95.0,25.0 Z" /></g>'
    )
