from . import unittest
from shapely.ops import substring
from shapely.geos import geos_version
from shapely.geometry import Point, LineString


class SubstringTestCase(unittest.TestCase):

    def setUp(self):
        self.point = Point(1, 1)
        self.line1 = LineString(([0, 0], [2, 0]))
        self.line2 = LineString(([3, 0], [3, 6], [4.5, 6]))
  
    @unittest.skipIf(geos_version < (3, 2, 0), 'GEOS 3.2.0 required')
    def test_return_startpoint(self):
        self.assertTrue(substring(self.line1, -500, -600).equals(Point(0, 0)))
        self.assertTrue(substring(self.line1, -500, -500).equals(Point(0, 0)))
        self.assertTrue(substring(self.line1, -1, -1.1, True).equals(Point(0, 0)))
        self.assertTrue(substring(self.line1, -1.1, -1.1, True).equals(Point(0, 0)))               

    @unittest.skipIf(geos_version < (3, 2, 0), 'GEOS 3.2.0 required')
    def test_return_endpoint(self):
        self.assertTrue(substring(self.line1, 500, 600).equals(Point(2, 0)))
        self.assertTrue(substring(self.line1, 500, 500).equals(Point(2, 0)))
        self.assertTrue(substring(self.line1, 1, 1.1, True).equals(Point(2, 0)))   
        self.assertTrue(substring(self.line1, 1.1, 1.1, True).equals(Point(2, 0)))

    @unittest.skipIf(geos_version < (3, 2, 0), 'GEOS 3.2.0 required')
    def test_return_midpoint(self):
        self.assertTrue(substring(self.line1, 0.5, 0.5).equals(Point(0.5, 0)))        
        self.assertTrue(substring(self.line1, -0.5, -0.5).equals(Point(1.5, 0)))
        self.assertTrue(substring(self.line1, 0.5, 0.5, True).equals(Point(1, 0)))        
        self.assertTrue(substring(self.line1, -0.5, -0.5, True).equals(Point(1, 0)))       

    @unittest.skipIf(geos_version < (3, 2, 0), 'GEOS 3.2.0 required')
    def test_return_startsubstring(self):
        self.assertTrue(substring(self.line1, -500, 0.6).equals(LineString(([0, 0], [0.6, 0]))))
        self.assertTrue(substring(self.line1, -1.1, 0.6, True).equals(LineString(([0, 0], [1.2, 0]))))

    @unittest.skipIf(geos_version < (3, 2, 0), 'GEOS 3.2.0 required')
    def test_return_endsubstring(self):
        self.assertTrue(substring(self.line1, 0.6, 500).equals(LineString(([0.6, 0], [2, 0]))))
        self.assertTrue(substring(self.line1, 0.6, 1.1, True).equals(LineString(([1.2, 0], [2, 0]))))

    @unittest.skipIf(geos_version < (3, 2, 0), 'GEOS 3.2.0 required')
    def test_return_midsubstring(self):
        self.assertTrue(substring(self.line1, 0.5, 0.6).equals(LineString(([0.5, 0], [0.6, 0]))))
        self.assertTrue(substring(self.line1, 0.6, 0.5).equals(LineString(([0.6, 0], [0.5, 0]))))
        self.assertTrue(substring(self.line1, -0.5, -0.6).equals(LineString(([1.5, 0], [1.4, 0]))))
        self.assertTrue(substring(self.line1, -0.6, -0.5).equals(LineString(([1.4, 0], [1.5, 0]))))
        self.assertTrue(substring(self.line1, 0.5, 0.6, True).equals(LineString(([1, 0], [1.2, 0]))))
        self.assertTrue(substring(self.line1, 0.6, 0.5, True).equals(LineString(([1.2, 0], [1, 0]))))
        self.assertTrue(substring(self.line1, -0.5, -0.6, True).equals(LineString(([1, 0], [0.8, 0]))))
        self.assertTrue(substring(self.line1, -0.6, -0.5, True).equals(LineString(([0.8, 0], [1, 0]))))

    @unittest.skipIf(geos_version < (3, 2, 0), 'GEOS 3.2.0 required')
    def test_return_midsubstring(self):
        self.assertTrue(substring(self.line1, 0.5, 0.6).equals(LineString(([0.5, 0], [0.6, 0]))))
        self.assertTrue(substring(self.line1, 0.6, 0.5).equals(LineString(([0.6, 0], [0.5, 0]))))
        self.assertTrue(substring(self.line1, -0.5, -0.6).equals(LineString(([1.5, 0], [1.4, 0]))))
        self.assertTrue(substring(self.line1, -0.6, -0.5).equals(LineString(([1.4, 0], [1.5, 0]))))
        self.assertTrue(substring(self.line1, 0.5, 0.6, True).equals(LineString(([1, 0], [1.2, 0]))))
        self.assertTrue(substring(self.line1, 0.6, 0.5, True).equals(LineString(([1.2, 0], [1, 0]))))
        self.assertTrue(substring(self.line1, -0.5, -0.6, True).equals(LineString(([1, 0], [0.8, 0]))))
        self.assertTrue(substring(self.line1, -0.6, -0.5, True).equals(LineString(([0.8, 0], [1, 0]))))

    @unittest.skipIf(geos_version < (3, 2, 0), 'GEOS 3.2.0 required')
    def test_return_substring_with_vertices(self):
        self.assertTrue(substring(self.line2, 1, 7).equals(LineString(([3, 1], [3, 6], [4, 6]))))
        self.assertTrue(substring(self.line2, 0.2, 0.9, True).equals(LineString(([3, 1.5], [3, 6], [3.75, 6]))))


def test_suite():
    loader = unittest.TestLoader()
    return loader.loadTestsFromTestCase(SubstringTestCase)
