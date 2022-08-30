from . import unittest
from shapely.geometry import CAP_STYLE, JOIN_STYLE


class StylesTest(unittest.TestCase):

    def test_cap(self):
        assert CAP_STYLE.round == 1
        assert CAP_STYLE.flat == 2
        assert CAP_STYLE.square == 3

    def test_join(self):
        assert JOIN_STYLE.round == 1
        assert JOIN_STYLE.mitre == 2
        assert JOIN_STYLE.bevel == 3
