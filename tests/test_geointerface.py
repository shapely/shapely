from . import unittest
from shapely.geometry import asShape, shape
from shapely.geometry.multipoint import MultiPoint
from shapely.geometry.linestring import LineString
from shapely.geometry.multilinestring import MultiLineString
from shapely.geometry.polygon import Polygon
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry.geo import _extract_crs


class GeoThing(object):
    def __init__(self, d):
        self.__geo_interface__ = d


class GeoInterfaceTestCase(unittest.TestCase):

    def _test_geointerface(self, shape_func):
        # Adapt a dictionary
        crs_obj = {"type": "name", "properties": {"name": "epsg:4326"}}
        d = {"type": "Point", "coordinates": (0.0, 0.0), 'crs': crs_obj}
        shape_obj = shape_func(d)
        self.assertEqual(shape_obj.geom_type, 'Point')
        self.assertEqual(tuple(shape_obj.coords), ((0.0, 0.0),))
        self.assertEqual(shape_obj.crs, crs_obj)
        self.assertEqual(shape_obj.__geo_interface__['crs'], crs_obj)

        # Adapt an object that implements the geo protocol
        thing = GeoThing({"type": "Point", "coordinates": (0.0, 0.0), 'crs': crs_obj})
        shape_obj = shape_func(thing)
        self.assertEqual(shape_obj.geom_type, 'Point')
        self.assertEqual(tuple(shape_obj.coords), ((0.0, 0.0),))
        self.assertEqual(shape_obj.crs, crs_obj)
        self.assertEqual(shape_obj.__geo_interface__['crs'], crs_obj)

        # Check line string
        shape_obj = shape_func(
            {'type': 'LineString', 'coordinates': ((-1.0, -1.0), (1.0, 1.0)), 'crs': crs_obj})
        self.assertIsInstance(shape_obj, LineString)
        self.assertEqual(tuple(shape_obj.coords), ((-1.0, -1.0), (1.0, 1.0)))
        self.assertEqual(shape_obj.crs, crs_obj)
        self.assertEqual(shape_obj.__geo_interface__['crs'], crs_obj)

        # polygon
        shape_obj = shape_func(
            {'type': 'Polygon',
             'coordinates':
                 (((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (2.0, -1.0), (0.0, 0.0)),
                  ((0.1, 0.1), (0.1, 0.2), (0.2, 0.2), (0.2, 0.1), (0.1, 0.1))),
             'crs': crs_obj
             }
        )
        self.assertIsInstance(shape_obj, Polygon)
        self.assertEqual(
            tuple(shape_obj.exterior.coords),
            ((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (2.0, -1.0), (0.0, 0.0)))
        self.assertEqual(len(shape_obj.interiors), 1)
        self.assertEqual(shape_obj.crs, crs_obj)
        self.assertEqual(shape_obj.__geo_interface__['crs'], crs_obj)

        # multi point
        shape_obj = shape_func({
            'type': 'MultiPoint',
            'coordinates': ((1.0, 2.0), (3.0, 4.0)),
            'crs': crs_obj
        })
        self.assertIsInstance(shape_obj, MultiPoint)
        self.assertEqual(len(shape_obj.geoms), 2)
        self.assertEqual(shape_obj.crs, crs_obj)
        self.assertEqual(shape_obj.__geo_interface__['crs'], crs_obj)

        # multi line string
        shape_obj = shape_func({
            'type': 'MultiLineString',
            'coordinates': (((0.0, 0.0), (1.0, 2.0)),),
            'crs': crs_obj
        })
        self.assertIsInstance(shape_obj, MultiLineString)
        self.assertEqual(len(shape_obj.geoms), 1)
        self.assertEqual(shape_obj.crs, crs_obj)
        self.assertEqual(shape_obj.__geo_interface__['crs'], crs_obj)

        # multi polygon
        shape_obj = shape_func(
            {'type': 'MultiPolygon',
             'coordinates':
                 [(((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)),
                   ((0.1, 0.1), (0.1, 0.2), (0.2, 0.2), (0.2, 0.1), (0.1, 0.1))
                   )],
             'crs': crs_obj
             })
        self.assertIsInstance(shape_obj, MultiPolygon)
        self.assertEqual(len(shape_obj.geoms), 1)
        self.assertEqual(shape_obj.crs, crs_obj)
        self.assertEqual(shape_obj.__geo_interface__['crs'], crs_obj)

        # unknown shape
        with self.assertRaises(ValueError):
            shape_func(
                {'type': 'MyAwesomePoly',
                 'coordinates':
                     [(((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)),
                       ((0.1, 0.1), (0.1, 0.2), (0.2, 0.2), (0.2, 0.1), (0.1, 0.1))
                       )],
                 'crs': crs_obj
                 })

    def test_geointerface(self):
        self._test_geointerface(shape)
        self._test_geointerface(asShape)

    def test_extract_crs(self):
        crs = 'my crs obj'
        thing = GeoThing({})
        thing.crs = crs
        self.assertEquals(crs, _extract_crs(thing))


def test_suite():
    return unittest.TestLoader().loadTestsFromTestCase(GeoInterfaceTestCase)
