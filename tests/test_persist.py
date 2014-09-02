"""Persistence tests
"""
from . import unittest
import pickle
from shapely import wkb, wkt
from shapely.geometry import Point


class PersistTestCase(unittest.TestCase):

    def test_pickle(self):

        p = Point(0.0, 0.0)
        data = pickle.dumps(p)
        q = pickle.loads(data)
        self.assertTrue(q.equals(p))

    def test_wkb(self):

        p = Point(0.0, 0.0)
        bytes = wkb.dumps(p)
        pb = wkb.loads(bytes)
        self.assertTrue(pb.equals(p))

    def test_wkt(self):
        p = Point(0.0, 0.0)
        text = wkt.dumps(p)
        self.assertTrue(text.startswith('POINT'))
        pt = wkt.loads(text)
        self.assertTrue(pt.equals(p))


def test_suite():
    return unittest.TestLoader().loadTestsFromTestCase(PersistTestCase)
