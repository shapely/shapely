import unittest
from shapely import geometry

class XYTestCase(unittest.TestCase):
    """New geometry/coordseq method 'xy' makes numpy interop easier"""
    def test_arrays(self):
        x, y = geometry.LineString(((0, 0), (1, 1))).xy
        self.failUnless(len(x) == 2)
        self.failUnless(list(x) == [0.0, 1.0])
        self.failUnless(len(y) == 2)
        self.failUnless(list(y) == [0.0, 1.0])

def test_suite():
    return unittest.TestLoader().loadTestsFromTestCase(XYTestCase)
