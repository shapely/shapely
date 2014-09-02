from . import unittest
from shapely.geometry import LineString
from shapely.geometry.collection import GeometryCollection


class CollectionTestCase(unittest.TestCase):

    def test_array_interface(self):
        m = GeometryCollection()
        self.assertEqual(len(m), 0)
        self.assertEqual(m.geoms, [])

    def test_child_with_deleted_parent(self):
        # test that we can remove a collection while having
        # childs around
        a = LineString([(0, 0), (1, 1), (1, 2), (2, 2)])
        b = LineString([(0, 0), (1, 1), (2, 1), (2, 2)])
        collection = a.intersection(b)

        child = collection.geoms[0]
        # delete parent of child
        del collection

        # access geometry, this should not seg fault as 1.2.15 did
        self.assertIsNotNone(child.wkt)


def test_suite():
    return unittest.TestLoader().loadTestsFromTestCase(CollectionTestCase)
