from shapely.geometry import LineString


class TestXY:
    """New geometry/coordseq method 'xy' makes numpy interop easier"""

    def test_arrays(self):
        x, y = LineString(((0, 0), (1, 1))).xy
        assert len(x) == 2
        assert list(x) == [0.0, 1.0]
        assert len(y) == 2
        assert list(y) == [0.0, 1.0]
