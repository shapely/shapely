import gc

from . import unittest

from shapely.strtree import STRtree
from shapely.geometry import Point, Polygon
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

    def test_insert_empty_geometry(self):
        """
        Passing nothing but empty geometries results in an empty strtree.
        The query segfaults if the empty geometry was actually inserted.
        """
        empty = Polygon()
        geoms = [empty]
        tree = STRtree(geoms)
        assert(tree._n_geoms == 0)
        query = Polygon([(0,0),(1,1),(2,0),(0,0)])
        results = tree.query(query)

    def test_query_empty_geometry(self):
        """
        Empty geometries should be filtered out.
        The query segfaults if the empty geometry was actually inserted.
        """
        empty = Polygon()
        point = Point(1, 0.5)
        geoms = [empty, point]
        tree = STRtree(geoms)
        assert(tree._n_geoms == 1)
        query = Polygon([(0,0),(1,1),(2,0),(0,0)])
        results = tree.query(query)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], point)

    def test_references(self):
        """Don't crash due to dangling references"""
        empty = Polygon()
        point = Point(1, 0.5)
        geoms = [empty, point]
        tree = STRtree(geoms)
        assert(tree._n_geoms == 1)

        empty = None
        point = None
        gc.collect()

        query = Polygon([(0,0),(1,1),(2,0),(0,0)])
        results = tree.query(query)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], Point(1, 0.5))


def test_suite():
    return unittest.TestLoader().loadTestsFromTestCase(STRTestCase)

if __name__ == '__main__':
    unittest.main()
