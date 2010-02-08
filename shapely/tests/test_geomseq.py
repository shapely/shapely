import unittest
from shapely import geometry

class MultiLineTestCase(unittest.TestCase):
    def test_array_interface(self):
        m = geometry.MultiLineString([((0, 0), (1, 1)), ((2, 2), (3, 3))])
        x = m.geoms[0].__array_interface__
        self.failUnless(
            repr(x['data']).find(
                '<shapely.coords.c_double_Array_4 object at 0x') == 0
                )

def test_suite():
    return unittest.TestLoader().loadTestsFromTestCase(MultiLineTestCase)
