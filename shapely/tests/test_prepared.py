import unittest
from shapely import prepared
from shapely import geometry


class PreparedGeometryTestCase(unittest.TestCase):
    
    def test_prepared(self):
        p = prepared.PreparedGeometry(geometry.Point(0.0, 0.0))


def test_suite():
    return unittest.TestLoader().loadTestsFromTestCase(
                                    PreparedGeometryTestCase
                                    )
