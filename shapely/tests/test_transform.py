import unittest
#from math import sin, cos, tan, pi
from shapely.algorithms import transform
from shapely.wkt import loads as load_wkt
from shapely.geometry import Point

class AffineTestCase(unittest.TestCase):
    def test_affine_params(self):
        g = load_wkt('LINESTRING(2.4 4.1, 2.4 3, 3 3)')
        self.assertRaises(TypeError, transform.affine, g, None)
        self.assertRaises(TypeError, transform.affine, g, '123456')
        self.assertRaises(ValueError, transform.affine, g, [1,2,3,4,5,6,7,8,9])
        self.assertRaises(AttributeError, transform.affine, None, [1,2,3,4,5,6])

    def test_affine_2d(self):
        g = load_wkt('LINESTRING(2.4 4.1, 2.4 3, 3 3)')
        # scale by different factors, and some transform too
        expected2d = load_wkt('LINESTRING(-0.2 14.35, -0.2 11.6, 1 11.6)')
        matrix2d = (2, 0,
                    0, 2.5,
                    -5, 4.1)
        a2 = transform.affine(g, matrix2d)
        self.assertTrue(a2.almost_equals(expected2d))
        self.assertFalse(a2.has_z)
        # Make sure a 3D matrix does not make a 3D shape from a 2D input
        matrix3d = (2, 0, 0,
                    0, 2.5, 0,
                    0, 0, 10,
                    -5, 4.1, 100)
        a3 = transform.affine(g, matrix3d)
        self.assertTrue(a3.almost_equals(expected2d))
        self.assertFalse(a3.has_z)

    def test_affine_3d(self):
        g2 = load_wkt('LINESTRING(2.4 4.1, 2.4 3, 3 3)')
        g3 = load_wkt('LINESTRING(2.4 4.1 100.2, 2.4 3 132.8, 3 3 128.6)')
        # scale by different factors
        matrix2d = (2, 0,
                    0, 2.5,
                    -5, 4.1)
        matrix3d = (2, 0, 0,
                    0, 2.5, 0,
                    0, 0, 0.3048,
                    -5, 4.1, 100)
        # All combinations of 2D and 3D
        a22 = transform.affine(g2, matrix2d)
        a23 = transform.affine(g2, matrix3d)
        a32 = transform.affine(g3, matrix2d)
        a33 = transform.affine(g3, matrix3d)
        # Check dimensions
        self.assertFalse(a22.has_z)
        self.assertFalse(a23.has_z)
        self.assertTrue(a32.has_z)
        self.assertTrue(a33.has_z)
        # 2D equality cheks
        expected2d = load_wkt('LINESTRING(-0.2 14.35, -0.2 11.6, 1 11.6)')
        expected3d = load_wkt('LINESTRING(-0.2 14.35 130.54096, '\
                              '-0.2 11.6 140.47744, 1 11.6 139.19728)')
        expected32 = load_wkt('LINESTRING(-0.2 14.35 100.2, '\
                              '-0.2 11.6 132.8, 1 11.6 128.6)')
        self.assertTrue(a22.almost_equals(expected2d))
        self.assertTrue(a33.almost_equals(expected3d))
        # Do explicit 3D check of coordinate values
        for a, e in zip(a32.coords, expected32.coords):
            for ap, ep in zip(a, e):
                self.assertAlmostEqual(ap, ep)
        for a, e in zip(a33.coords, expected3d.coords):
            for ap, ep in zip(a, e):
                self.assertAlmostEqual(ap, ep)

# TODO: finish these!
class RotateTestCase(unittest.TestCase):
    pass

class ScaleTestCase(unittest.TestCase):
    pass

class SkewTestCase(unittest.TestCase):
    pass

class TranslateTestCase(unittest.TestCase):
    pass

def test_suite():
    loader = unittest.TestLoader()
    return unittest.TestSuite([
        unittest.TestLoader().loadTestsFromTestCase(AffineTestCase),
        unittest.TestLoader().loadTestsFromTestCase(RotateTestCase),
        unittest.TestLoader().loadTestsFromTestCase(ScaleTestCase),
        unittest.TestLoader().loadTestsFromTestCase(SkewTestCase),
        unittest.TestLoader().loadTestsFromTestCase(TranslateTestCase)])
