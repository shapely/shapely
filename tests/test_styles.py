from . import unittest
from shapely.geometry import CAP_STYLE, JOIN_STYLE
from shapely.constructive import BufferCapStyles, BufferJoinStyles


class StylesTest(unittest.TestCase):
    def test_cap(self):
        self.assertEqual(CAP_STYLE.round, 1)
        self.assertEqual(CAP_STYLE.round, BufferCapStyles.round)
        self.assertEqual(CAP_STYLE.flat, 2)
        self.assertEqual(CAP_STYLE.flat, BufferCapStyles.flat)
        self.assertEqual(CAP_STYLE.square, 3)
        self.assertEqual(CAP_STYLE.square, BufferCapStyles.square)

    def test_join(self):
        self.assertEqual(JOIN_STYLE.round, 1)
        self.assertEqual(JOIN_STYLE.round, BufferJoinStyles.round)
        self.assertEqual(JOIN_STYLE.mitre, 2)
        self.assertEqual(JOIN_STYLE.mitre, BufferJoinStyles.mitre)
        self.assertEqual(JOIN_STYLE.bevel, 3)
        self.assertEqual(JOIN_STYLE.bevel, BufferJoinStyles.bevel)
