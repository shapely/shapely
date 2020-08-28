from . import unittest, shapely20_deprecated

import pytest

from shapely.geometry import shape
from shapely.geometry.multipoint import MultiPoint
from shapely.geometry.linestring import LineString
from shapely.geometry.multilinestring import MultiLineString
from shapely.geometry.polygon import Polygon
from shapely.geometry.multipolygon import MultiPolygon
from shapely import wkt


class GeoThing(object):
    def __init__(self, d):
        self.__geo_interface__ = d


class GeoInterfaceTestCase(unittest.TestCase):

    def test_geointerface(self):
        # Convert a dictionary
        d = {"type": "Point", "coordinates": (0.0, 0.0)}
        geom = shape(d)
        self.assertEqual(geom.geom_type, 'Point')
        self.assertEqual(tuple(geom.coords), ((0.0, 0.0),))

        # Convert an object that implements the geo protocol
        geom = None
        thing = GeoThing({"type": "Point", "coordinates": (0.0, 0.0)})
        geom = shape(thing)
        self.assertEqual(geom.geom_type, 'Point')
        self.assertEqual(tuple(geom.coords), ((0.0, 0.0),))

        # Check line string
        geom = shape(
            {'type': 'LineString', 'coordinates': ((-1.0, -1.0), (1.0, 1.0))})
        self.assertIsInstance(geom, LineString)
        self.assertEqual(tuple(geom.coords), ((-1.0, -1.0), (1.0, 1.0)))

        # polygon
        geom = shape(
            {'type': 'Polygon',
             'coordinates':
                (((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (2.0, -1.0), (0.0, 0.0)),
                 ((0.1, 0.1), (0.1, 0.2), (0.2, 0.2), (0.2, 0.1), (0.1, 0.1)))}
        )
        self.assertIsInstance(geom, Polygon)
        self.assertEqual(
            tuple(geom.exterior.coords),
            ((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (2.0, -1.0), (0.0, 0.0)))
        self.assertEqual(len(geom.interiors), 1)

        # multi point
        geom = shape({'type': 'MultiPoint',
                       'coordinates': ((1.0, 2.0), (3.0, 4.0))})
        self.assertIsInstance(geom, MultiPoint)
        self.assertEqual(len(geom.geoms), 2)

        # multi line string
        geom = shape({'type': 'MultiLineString',
                       'coordinates': (((0.0, 0.0), (1.0, 2.0)),)})
        self.assertIsInstance(geom, MultiLineString)
        self.assertEqual(len(geom.geoms), 1)

        # multi polygon
        geom = shape(
            {'type': 'MultiPolygon',
             'coordinates':
                [(((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)),
                  ((0.1, 0.1), (0.1, 0.2), (0.2, 0.2), (0.2, 0.1), (0.1, 0.1))
                  )]})
        self.assertIsInstance(geom, MultiPolygon)
        self.assertEqual(len(geom.geoms), 1)


def test_empty_wkt_polygon():
    """Confirm fix for issue #450"""
    g = wkt.loads('POLYGON EMPTY')
    assert g.__geo_interface__['type'] == 'Polygon'
    assert g.__geo_interface__['coordinates'] == ()


def test_empty_polygon():
    """Confirm fix for issue #450"""
    g = Polygon()
    assert g.__geo_interface__['type'] == 'Polygon'
    assert g.__geo_interface__['coordinates'] == ()
