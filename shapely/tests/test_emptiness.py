import unittest
from shapely.geometry.base import BaseGeometry
from shapely.geometry import Point

class EmptinessTestCase(unittest.TestCase):
    def test_empty_base(self):
        g = BaseGeometry()
        self.failUnless(g._is_empty, True)
    def test_empty_point(self):
        p = Point()
        self.failUnless(p._is_empty, True)
    def test_emptying_point(self):
        p = Point(0, 0)
        self.failIf(p._is_empty, False)
        p.empty()
        self.failUnless(p._is_empty, True)

def test_suite():
    return unittest.TestLoader().loadTestsFromTestCase(EmptinessTestCase)
