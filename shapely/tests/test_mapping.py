import unittest
from shapely.geometry import Point, mapping

class MappingTestCase(unittest.TestCase):
    def test_point(self):
        m = mapping(Point(0, 0))
        self.failUnlessEqual(m['type'], 'Point')
        self.failUnlessEqual(m['coordinates'], (0.0, 0.0))

def test_suite():
    return unittest.TestLoader().loadTestsFromTestCase(MappingTestCase)
