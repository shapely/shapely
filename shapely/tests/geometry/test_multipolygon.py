import numpy as np
import pytest

from shapely import (
    GeometryCollection,
    LinearRing,
    LineString,
    MultiPolygon,
    Point,
    Polygon,
)
from shapely.geometry.base import dump_coords
from shapely.tests.geometry.test_multi import MultiGeometryTestCase


class TestMultiPolygon(MultiGeometryTestCase):
    def test_multipolygon(self):
        # Empty
        geom = MultiPolygon([])
        assert geom.is_empty
        assert len(geom.geoms) == 0

        # From coordinate tuples
        coords = [
            (
                ((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)),
                [((0.25, 0.25), (0.25, 0.5), (0.5, 0.5), (0.5, 0.25))],
            )
        ]
        geom = MultiPolygon(coords)
        assert isinstance(geom, MultiPolygon)
        assert len(geom.geoms) == 1
        assert dump_coords(geom) == [
            [
                (0.0, 0.0),
                (0.0, 1.0),
                (1.0, 1.0),
                (1.0, 0.0),
                (0.0, 0.0),
                [(0.25, 0.25), (0.25, 0.5), (0.5, 0.5), (0.5, 0.25), (0.25, 0.25)],
            ]
        ]

        # Or without holes
        coords2 = [(((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)),)]
        geom = MultiPolygon(coords2)
        assert isinstance(geom, MultiPolygon)
        assert len(geom.geoms) == 1
        assert dump_coords(geom) == [
            [
                (0.0, 0.0),
                (0.0, 1.0),
                (1.0, 1.0),
                (1.0, 0.0),
                (0.0, 0.0),
            ]
        ]

        # Or from polygons
        p = Polygon(
            ((0, 0), (0, 1), (1, 1), (1, 0)),
            [((0.25, 0.25), (0.25, 0.5), (0.5, 0.5), (0.5, 0.25))],
        )
        geom = MultiPolygon([p])
        assert len(geom.geoms) == 1
        assert dump_coords(geom) == [
            [
                (0.0, 0.0),
                (0.0, 1.0),
                (1.0, 1.0),
                (1.0, 0.0),
                (0.0, 0.0),
                [(0.25, 0.25), (0.25, 0.5), (0.5, 0.5), (0.5, 0.25), (0.25, 0.25)],
            ]
        ]

        # None and empty polygons are dropped
        geom_from_list_with_empty = MultiPolygon([p, None, Polygon()])
        assert geom_from_list_with_empty == geom

        # Or from a list of multiple polygons
        geom_multiple_from_list = MultiPolygon([p, p])
        assert len(geom_multiple_from_list.geoms) == 2
        assert all(p == geom.geoms[0] for p in geom_multiple_from_list.geoms)

        # Or from a np.array of polygons
        geom_multiple_from_array = MultiPolygon(np.array([p, p]))
        assert geom_multiple_from_array == geom_multiple_from_list

        # Or from another multi-polygon
        geom2 = MultiPolygon(geom)
        assert len(geom2.geoms) == 1
        assert dump_coords(geom2) == [
            [
                (0.0, 0.0),
                (0.0, 1.0),
                (1.0, 1.0),
                (1.0, 0.0),
                (0.0, 0.0),
                [(0.25, 0.25), (0.25, 0.5), (0.5, 0.5), (0.5, 0.25), (0.25, 0.25)],
            ]
        ]

        # Sub-geometry Access
        assert isinstance(geom.geoms[0], Polygon)
        assert dump_coords(geom.geoms[0]) == [
            (0.0, 0.0),
            (0.0, 1.0),
            (1.0, 1.0),
            (1.0, 0.0),
            (0.0, 0.0),
            [(0.25, 0.25), (0.25, 0.5), (0.5, 0.5), (0.5, 0.25), (0.25, 0.25)],
        ]
        with pytest.raises(IndexError):  # index out of range
            geom.geoms[1]

        # Geo interface
        assert geom.__geo_interface__ == {
            "type": "MultiPolygon",
            "coordinates": [
                (
                    ((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)),
                    ((0.25, 0.25), (0.25, 0.5), (0.5, 0.5), (0.5, 0.25), (0.25, 0.25)),
                )
            ],
        }

    def test_subgeom_access(self):
        poly0 = Polygon([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)])
        poly1 = Polygon([(0.25, 0.25), (0.25, 0.5), (0.5, 0.5), (0.5, 0.25)])
        self.subgeom_access_test(MultiPolygon, [poly0, poly1])


def test_fail_list_of_multipolygons():
    """A list of multipolygons is not a valid multipolygon ctor argument"""
    poly = Polygon([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)])
    multi = MultiPolygon(
        [
            (
                ((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)),
                [((0.25, 0.25), (0.25, 0.5), (0.5, 0.5), (0.5, 0.25))],
            )
        ]
    )
    with pytest.raises(ValueError):
        MultiPolygon([multi])

    with pytest.raises(ValueError):
        MultiPolygon([poly, multi])


@pytest.mark.parametrize(
    "geom",
    [
        GeometryCollection([Polygon([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)])]),
        Point(0, 0),
        LineString([(0, 0), (1, 1)]),
        LinearRing([(0, 0), (1, 1), (1, 0)]),
    ],
    ids=["GeometryCollection", "Point", "LineString", "LinearRing"],
)
def test_fail_list_with_non_polygon_geometry(geom):
    """A non-Polygon geometry in the list raises a clear ValueError.

    Previously a non-Polygon geometry fell through to ``shell = ob[0]`` and
    raised a cryptic ``TypeError: ... object is not subscriptable``. It is now
    rejected with a ValueError naming the offending type, regardless of its
    position in the list (#2178).
    """
    poly = Polygon([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)])

    # As the only element, before a valid Polygon, and after a valid Polygon.
    for inputs in ([geom], [geom, poly], [poly, geom]):
        with pytest.raises(ValueError, match=geom.geom_type):
            MultiPolygon(inputs)


def test_numpy_object_array():
    geom = MultiPolygon(
        [
            (
                ((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)),
                [((0.25, 0.25), (0.25, 0.5), (0.5, 0.5), (0.5, 0.25))],
            )
        ]
    )
    ar = np.empty(1, object)
    ar[:] = [geom]
    assert ar[0] == geom
