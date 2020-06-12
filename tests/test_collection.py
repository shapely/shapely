from . import unittest, shapely20_deprecated

from shapely.errors import ShapelyDeprecationWarning
from shapely.geometry import LineString
from shapely.geometry.collection import GeometryCollection
from shapely.geometry import shape
from shapely.geometry import asShape

import pytest


class CollectionTestCase(unittest.TestCase):

    def test_array_interface(self):
        m = GeometryCollection()
        self.assertEqual(len(m), 0)
        self.assertEqual(m.geoms, [])

    def test_child_with_deleted_parent(self):
        # test that we can remove a collection while having
        # childs around
        a = LineString([(0, 0), (1, 1), (1, 2), (2, 2)])
        b = LineString([(0, 0), (1, 1), (2, 1), (2, 2)])
        collection = a.intersection(b)

        child = collection.geoms[0]
        # delete parent of child
        del collection

        # access geometry, this should not seg fault as 1.2.15 did
        self.assertIsNotNone(child.wkt)

    @shapely20_deprecated
    def test_geointerface_adapter(self):
        d = {"type": "GeometryCollection","geometries": [
                {"type": "Point", "coordinates": (0, 3)},
                {"type": "LineString", "coordinates": ((2, 0), (1, 0))}
            ]}

        # asShape
        m = asShape(d)
        self.assertEqual(m.geom_type, "GeometryCollection")
        self.assertEqual(len(m), 2)
        geom_types = [g.geom_type for g in m.geoms]
        self.assertIn("Point", geom_types)
        self.assertIn("LineString", geom_types)

    def test_geointerface(self):
        d = {"type": "GeometryCollection","geometries": [
                {"type": "Point", "coordinates": (0, 3)},
                {"type": "LineString", "coordinates": ((2, 0), (1, 0))}
            ]}

        # shape
        m = shape(d)
        self.assertEqual(m.geom_type, "GeometryCollection")
        self.assertEqual(len(m), 2)
        geom_types = [g.geom_type for g in m.geoms]
        self.assertIn("Point", geom_types)
        self.assertIn("LineString", geom_types)

    @shapely20_deprecated
    def test_empty_geointerface_adapter(self):
        d = {"type": "GeometryCollection", "geometries": []}

        # asShape
        m = asShape(d)
        self.assertEqual(m.geom_type, "GeometryCollection")
        self.assertEqual(len(m), 0)
        self.assertEqual(m.geoms, [])

    def test_empty_geointerface(self):
        d = {"type": "GeometryCollection", "geometries": []}

        # shape
        m = shape(d)
        self.assertEqual(m.geom_type, "GeometryCollection")
        self.assertEqual(len(m), 0)
        self.assertEqual(m.geoms, [])

    def test_empty_coordinates(self):

        d = {"type": "GeometryCollection", "geometries": [
            {"type": "Point", "coordinates": ()},
            {"type": "LineString", "coordinates": (())}
        ]}

        # shape
        m = shape(d)
        self.assertEqual(m.geom_type, "GeometryCollection")
        self.assertEqual(len(m), 0)
        self.assertEqual(m.geoms, [])


def test_geometrycollection_adapter_deprecated():
    d = {"type": "GeometryCollection","geometries": [
            {"type": "Point", "coordinates": (0, 3)},
            {"type": "LineString", "coordinates": ((2, 0), (1, 0))}
    ]}
    with pytest.warns(ShapelyDeprecationWarning):
        asShape(d)

    d = {"type": "GeometryCollection", "geometries": []}
    with pytest.warns(ShapelyDeprecationWarning):
        asShape(d)


def test_suite():
    return unittest.TestLoader().loadTestsFromTestCase(CollectionTestCase)
