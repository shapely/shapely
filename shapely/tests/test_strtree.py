from . import unittest

from shapely.strtree import STRtree
from shapely.geometry import Point
from shapely.geos import geos_version


@unittest.skipIf(geos_version < (3, 4, 2), 'GEOS 3.4.2 required')
class STRTestCase(unittest.TestCase):
    def test_query(self):
        points = [Point(i, i) for i in range(10)]
        tree = STRtree(points)
        results = tree.query(Point(2,2).buffer(0.99))
        self.assertEqual(len(results), 1)
        results = tree.query(Point(2,2).buffer(1.0))
        self.assertEqual(len(results), 3)


def test_suite():
    return unittest.TestLoader().loadTestsFromTestCase(STRTestCase)

if __name__ == '__main__':
    unittest.main()
