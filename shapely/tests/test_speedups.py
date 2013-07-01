import unittest

from shapely import speedups
from shapely.geometry import LineString, Polygon

class SpeedupsTestCase(unittest.TestCase):
    
    def setUp(self):
        self.assertFalse(speedups._orig)
        speedups.enable()
        self.assertTrue(speedups._orig)
        
    def tearDown(self):
        self.assertTrue(speedups._orig)
        speedups.disable()
        self.assertFalse(speedups._orig)

    def test_create_linestring(self):
        ls = LineString([(0, 0), (1, 0), (1, 2)])
        self.assertEqual(ls.length, 3)

    def test_create_polygon(self):
        p = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        self.assertEqual(p.length, 8)

    def test_create_polygon_from_linestring(self):
        ls = LineString([(0, 0), (2, 0), (2, 2), (0, 2)])
        p = Polygon(ls)
        self.assertEqual(p.length, 8)


def test_suite():
    if speedups.available:
        return unittest.TestLoader().loadTestsFromTestCase(
            SpeedupsTestCase
        )
    else:
        return lambda x: None