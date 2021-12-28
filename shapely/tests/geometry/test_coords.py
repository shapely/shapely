import numpy as np

from shapely.geometry import LineString


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
