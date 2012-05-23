# Tests of support for Numpy ndarrays. See
# https://github.com/sgillies/shapely/issues/26 for discussion.
# Requires numpy.

import unittest
from shapely import geometry
import numpy

class TransposeTestCase(unittest.TestCase):
    def test_multipoint(self):
        a = numpy.array([[1.0, 1.0, 2.0, 2.0, 1.0], [3.0, 4.0, 4.0, 3.0, 3.0]])
        t = a.T
        s = geometry.asMultiPoint(t)
        coords = reduce(lambda x, y: x + y, [list(g.coords) for g in s])
        self.failUnlessEqual(
            coords,
            [(1.0, 3.0), (1.0, 4.0), (2.0, 4.0), (2.0, 3.0), (1.0, 3.0)] )
    def test_linestring(self):
        a = numpy.array([[1.0, 1.0, 2.0, 2.0, 1.0], [3.0, 4.0, 4.0, 3.0, 3.0]])
        t = a.T
        s = geometry.asLineString(t)
        self.failUnlessEqual(
            list(s.coords),
            [(1.0, 3.0), (1.0, 4.0), (2.0, 4.0), (2.0, 3.0), (1.0, 3.0)] )
    def test_polygon(self):
        a = numpy.array([[1.0, 1.0, 2.0, 2.0, 1.0], [3.0, 4.0, 4.0, 3.0, 3.0]])
        t = a.T
        s = geometry.asPolygon(t)
        self.failUnlessEqual(
            list(s.exterior.coords),
            [(1.0, 3.0), (1.0, 4.0), (2.0, 4.0), (2.0, 3.0), (1.0, 3.0)] )

def test_suite():
    return unittest.TestLoader().loadTestsFromTestCase(TransposeTestCase)

