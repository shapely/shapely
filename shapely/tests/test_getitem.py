import unittest
from shapely import geometry

class CoordsGetItemTestCase(unittest.TestCase):
    def test_index_coords(self):
        c = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]
        g = geometry.LineString(c)
        self.assertTrue(g.coords[0] == c[0])
        self.assertTrue(g.coords[1] == c[1])
        self.assertTrue(g.coords[2] == c[2])
        self.assertRaises(IndexError, lambda: g.coords[3])
        self.assertTrue(g.coords[-1] == c[2])
        self.assertTrue(g.coords[-2] == c[1])
        self.assertTrue(g.coords[-3] == c[0])
        self.assertRaises(IndexError, lambda: g.coords[-4])

    def test_index_empty_coords(self):
        g = geometry.LineString()
        self.assertRaises(IndexError, lambda: g.coords[0])

    def test_slice_coords(self):
        c = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]
        g = geometry.LineString(c)
        self.assertTrue(g.coords[1:] == c[1:])
        self.assertTrue(g.coords[:-1] == c[:-1])
        self.assertTrue(g.coords[::-1] == c[::-1])
        self.assertTrue(g.coords[::2] == c[::2])
        self.assertTrue(g.coords[:4] == c[:4])
        self.assertTrue(g.coords[4:] == c[4:])

class MultiGeomGetItemTestCase(unittest.TestCase):
    def test_index_multigeom(self):
        c = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]
        g = geometry.MultiPoint(c)
        self.assertTrue(g[0].equals(geometry.Point(c[0])))
        self.assertTrue(g[1].equals(geometry.Point(c[1])))
        self.assertTrue(g[2].equals(geometry.Point(c[2])))
        self.assertRaises(IndexError, lambda: g[3])
        self.assertTrue(g[-1].equals(geometry.Point(c[-1])))
        self.assertTrue(g[-2].equals(geometry.Point(c[-2])))
        self.assertTrue(g[-3].equals(geometry.Point(c[-3])))
        self.assertRaises(IndexError, lambda: g[-4])

    def test_index_empty_coords(self):
        g = geometry.MultiLineString()
        self.assertRaises(IndexError, lambda: g[0])

    def test_slice_multigeom(self):
        c = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]
        g = geometry.MultiPoint(c)
        self.assertTrue(geometry.MultiPoint(g[1:]).equals(geometry.MultiPoint(c[1:])))
        self.assertTrue(geometry.MultiPoint(g[:-1]).equals(geometry.MultiPoint(c[:-1])))
        self.assertTrue(geometry.MultiPoint(g[::-1]).equals(geometry.MultiPoint(c[::-1])))
        self.assertTrue(geometry.MultiPoint(g[::2]).equals(geometry.MultiPoint(c[::2])))
        self.assertTrue(geometry.MultiPoint(g[:4]).equals(geometry.MultiPoint(c[:4])))
        self.assertTrue(geometry.MultiPoint(g[4:]).is_empty)

class LinearRingGetItemTestCase(unittest.TestCase):
    def test_index_linearring(self):
        shell = geometry.polygon.LinearRing([(0.0, 0.0), (70.0, 120.0), (140.0, 0.0), (0.0, 0.0)])
        holes = [geometry.polygon.LinearRing([(60.0, 80.0), (80.0, 80.0), (70.0, 60.0), (60.0, 80.0)]),
                 geometry.polygon.LinearRing([(30.0, 10.0), (50.0, 10.0), (40.0, 30.0), (30.0, 10.0)]),
                 geometry.polygon.LinearRing([(90.0, 10), (110.0, 10.0), (100.0, 30.0), (90.0, 10.0)])]
        g = geometry.Polygon(shell, holes)
        self.assertTrue(g.interiors[0].equals(holes[0]))
        self.assertTrue(g.interiors[1].equals(holes[1]))
        self.assertTrue(g.interiors[2].equals(holes[2]))
        self.assertRaises(IndexError, lambda: g.interiors[3].is_valid)
        self.assertTrue(g.interiors[-1].equals(holes[-1]))
        self.assertTrue(g.interiors[-2].equals(holes[-2]))
        self.assertTrue(g.interiors[-3].equals(holes[-3]))
        self.assertRaises(IndexError, lambda: g.interiors[-4].is_valid)

    def test_index_empty_linearring(self):
        g = geometry.Polygon()
        self.assertRaises(IndexError, lambda: g.interiors[0])

    def test_slice_linearring(self):
        shell = geometry.polygon.LinearRing([(0.0, 0.0), (70.0, 120.0), (140.0, 0.0), (0.0, 0.0)])
        holes = [geometry.polygon.LinearRing([(60.0, 80.0), (80.0, 80.0), (70.0, 60.0), (60.0, 80.0)]),
                 geometry.polygon.LinearRing([(30.0, 10.0), (50.0, 10.0), (40.0, 30.0), (30.0, 10.0)]),
                 geometry.polygon.LinearRing([(90.0, 10), (110.0, 10.0), (100.0, 30.0), (90.0, 10.0)])]
        g = geometry.Polygon(shell, holes)
        self.assertTrue(all([a.equals(b) for (a, b) in zip(g.interiors[1:], holes[1:])]))
        self.assertTrue(all([a.equals(b) for (a, b) in zip(g.interiors[:-1], holes[:-1])]))
        self.assertTrue(all([a.equals(b) for (a, b) in zip(g.interiors[::-1], holes[::-1])]))
        self.assertTrue(all([a.equals(b) for (a, b) in zip(g.interiors[::2], holes[::2])]))
        self.assertTrue(all([a.equals(b) for (a, b) in zip(g.interiors[:4], holes[:4])]))
        self.assertTrue(g.interiors[4:] == [])

def test_suite():
    loader = unittest.TestLoader()
    return unittest.TestSuite([
        unittest.TestLoader().loadTestsFromTestCase(CoordsGetItemTestCase),
        unittest.TestLoader().loadTestsFromTestCase(MultiGeomGetItemTestCase),
        unittest.TestLoader().loadTestsFromTestCase(LinearRingGetItemTestCase)])
