import unittest
from shapely.geometry.collection import GeometryCollection

class CollectionTestCase(unittest.TestCase):
    def test_array_interface(self):
        m = GeometryCollection()
        self.failUnlessEqual(len(m), 0)

def test_suite():
    return unittest.TestLoader().loadTestsFromTestCase(CollectionTestCase)
