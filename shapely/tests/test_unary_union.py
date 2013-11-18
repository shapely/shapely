from . import unittest
from shapely.geos import geos_version
from itertools import islice
from shapely.geometry import MultiPolygon
from shapely.geometry import Point
from shapely.ops import unary_union


def halton(base):
    """Returns an iterator over an infinite Halton sequence"""
    def value(index):
        result = 0.0
        f = 1.0 / base
        i = index
        while i > 0:
            result += f * (i % base)
            i = i // base
            f = f / base
        return result
    i = 1
    while i > 0:
        yield value(i)
        i += 1


class UnionTestCase(unittest.TestCase):
    # Instead of random points, use deterministic, pseudo-random Halton
    # sequences for repeatability sake.
    coords = list(zip(
        list(islice(halton(5), 20, 120)),
        list(islice(halton(7), 20, 120)),
    ))

    @unittest.skipIf(geos_version < (3, 3, 0), 'GEOS 3.3.0 required')
    def test_1(self):
        patches = [Point(xy).buffer(0.05) for xy in self.coords]
        u = unary_union(patches)
        self.assertEqual(u.geom_type, 'MultiPolygon')
        self.assertAlmostEqual(u.area, 0.71857254056)

    @unittest.skipIf(geos_version < (3, 3, 0), 'GEOS 3.3.0 required')
    def test_multi(self):
        # Test of multipart input based on comment by @schwehr at
        # https://github.com/Toblerity/Shapely/issues/47#issuecomment-21809308
        patches = MultiPolygon([Point(xy).buffer(0.05) for xy in self.coords])
        self.assertAlmostEqual(unary_union(patches).area,
                               0.71857254056)
        self.assertAlmostEqual(unary_union([patches, patches]).area,
                               0.71857254056)


def test_suite():
    return unittest.TestLoader().loadTestsFromTestCase(UnionTestCase)
