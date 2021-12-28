import pytest

from shapely.geometry import LinearRing, LineString, Polygon


class TestCoordsGetItem:
    def test_index_2d_coords(self):
        c = [(float(x), float(-x)) for x in range(4)]
        g = LineString(c)
        for i in range(-4, 4):
            assert g.coords[i] == c[i]
        with pytest.raises(IndexError):
            g.coords[4]
        with pytest.raises(IndexError):
            g.coords[-5]

    def test_index_3d_coords(self):
        c = [(float(x), float(-x), float(x * 2)) for x in range(4)]
        g = LineString(c)
        for i in range(-4, 4):
            assert g.coords[i] == c[i]
        with pytest.raises(IndexError):
            g.coords[4]
        with pytest.raises(IndexError):
            g.coords[-5]

    def test_index_coords_misc(self):
        g = LineString()  # empty
        with pytest.raises(IndexError):
            g.coords[0]
        with pytest.raises(TypeError):
            g.coords[0.0]

    def test_slice_2d_coords(self):
        c = [(float(x), float(-x)) for x in range(4)]
        g = LineString(c)
        assert g.coords[1:] == c[1:]
        assert g.coords[:-1] == c[:-1]
        assert g.coords[::-1] == c[::-1]
        assert g.coords[::2] == c[::2]
        assert g.coords[:4] == c[:4]
        assert g.coords[4:] == c[4:] == []

    def test_slice_3d_coords(self):
        c = [(float(x), float(-x), float(x * 2)) for x in range(4)]
        g = LineString(c)
        assert g.coords[1:] == c[1:]
        assert g.coords[:-1] == c[:-1]
        assert g.coords[::-1] == c[::-1]
        assert g.coords[::2] == c[::2]
        assert g.coords[:4] == c[:4]
        assert g.coords[4:] == c[4:] == []


class TestLinearRingGetItem:
    def test_index_linearring(self):
        shell = LinearRing([(0.0, 0.0), (70.0, 120.0), (140.0, 0.0), (0.0, 0.0)])
        holes = [
            LinearRing([(60.0, 80.0), (80.0, 80.0), (70.0, 60.0), (60.0, 80.0)]),
            LinearRing([(30.0, 10.0), (50.0, 10.0), (40.0, 30.0), (30.0, 10.0)]),
            LinearRing([(90.0, 10), (110.0, 10.0), (100.0, 30.0), (90.0, 10.0)]),
        ]
        g = Polygon(shell, holes)
        for i in range(-3, 3):
            assert g.interiors[i].equals(holes[i])
        with pytest.raises(IndexError):
            g.interiors[3]
        with pytest.raises(IndexError):
            g.interiors[-4]

    def test_index_linearring_misc(self):
        g = Polygon()  # empty
        with pytest.raises(IndexError):
            g.interiors[0]
        with pytest.raises(TypeError):
            g.interiors[0.0]

    def test_slice_linearring(self):
        shell = LinearRing([(0.0, 0.0), (70.0, 120.0), (140.0, 0.0), (0.0, 0.0)])
        holes = [
            LinearRing([(60.0, 80.0), (80.0, 80.0), (70.0, 60.0), (60.0, 80.0)]),
            LinearRing([(30.0, 10.0), (50.0, 10.0), (40.0, 30.0), (30.0, 10.0)]),
            LinearRing([(90.0, 10), (110.0, 10.0), (100.0, 30.0), (90.0, 10.0)]),
        ]
        g = Polygon(shell, holes)
        t = [a.equals(b) for (a, b) in zip(g.interiors[1:], holes[1:])]
        assert all(t)
        t = [a.equals(b) for (a, b) in zip(g.interiors[:-1], holes[:-1])]
        assert all(t)
        t = [a.equals(b) for (a, b) in zip(g.interiors[::-1], holes[::-1])]
        assert all(t)
        t = [a.equals(b) for (a, b) in zip(g.interiors[::2], holes[::2])]
        assert all(t)
        t = [a.equals(b) for (a, b) in zip(g.interiors[:3], holes[:3])]
        assert all(t)
        assert g.interiors[3:] == holes[3:] == []
