import numpy as np
import pytest

from shapely import LineString


class TestCoords:
    """
    Shapely assumes contiguous C-order float64 data for internal ops.
    Data should be converted to contiguous float64 if numpy exists.
    c9a0707 broke this a little bit.
    """

    def test_data_promotion(self):
        coords = np.array([[12, 34], [56, 78]], dtype=np.float32)
        processed_coords = np.array(LineString(coords).coords)

        assert coords.tolist() == processed_coords.tolist()

    def test_data_destriding(self):
        coords = np.array([[12, 34], [56, 78]], dtype=np.float32)

        # Easy way to introduce striding: reverse list order
        processed_coords = np.array(LineString(coords[::-1]).coords)

        assert coords[::-1].tolist() == processed_coords.tolist()


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


class TestXY:
    """New geometry/coordseq method 'xy' makes numpy interop easier"""

    def test_arrays(self):
        x, y = LineString([(0, 0), (1, 1)]).xy
        assert len(x) == 2
        assert list(x) == [0.0, 1.0]
        assert len(y) == 2
        assert list(y) == [0.0, 1.0]
