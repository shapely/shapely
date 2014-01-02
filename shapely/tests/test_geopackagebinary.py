"""GeoPackage persistence tests
"""
from . import unittest
from shapely import geopackagebinary
from shapely.geometry import Point, LineString


class GeoPackagePersistTestCase(unittest.TestCase):

    def test_gpb_round_trip(self):
        p = Point(1.0, 2.0)
        bytes = geopackagebinary.dumps(p)
        pb = geopackagebinary.loads(bytes)
        self.assertTrue(pb.equals(p))

    def test_gpb_out(self):
        p = Point(136, -35)
        bytes = geopackagebinary.dumps(p, True)
        self.assertEqual(bytes, "47500001000000000101000000000000000000614000000000008041C0")
    
    def test_gpb_in(self):
        pb = geopackagebinary.loads("47500001000000000101000000000000000000614000000000008041C0", True)
        self.assertTrue(pb.equals(Point(136, -35)))

    def test_gpb_in_envelope2d(self):
        pb = geopackagebinary.loads("47500003000000000000000000006140000000000000614000000000008041C000000000008041C00101000000000000000000614000000000008041C0", True)
        self.assertTrue(pb.equals(Point(136, -35)))

    def test_gpb_in_linestring(self):
        lsb = geopackagebinary.loads("47500003E61000000000000000C06040000000000000614000000000008041C000000000000040C0010200000003000000000000000000614000000000008041C00000000000C0604000000000008040C00000000000E0604000000000000040C0", True)
        ls = LineString(((136, -35), (134, -33), (135, -32)))
        self.assertTrue(lsb.equals(ls))


def test_suite():
    return unittest.TestLoader().loadTestsFromTestCase(GeoPackagePersistTestCase)
