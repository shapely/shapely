from . import unittest, numpy

import pytest

from shapely import geometry
from shapely.errors import ShapelyDeprecationWarning

class CoordsTestCase(unittest.TestCase):
    """
    Shapely assumes contiguous C-order float64 data for internal ops.
    Data should be converted to contiguous float64 if numpy exists.
    c9a0707 broke this a little bit.
    """

    @unittest.skipIf(not numpy, 'Numpy required')
    def test_data_promotion(self):
        coords = numpy.array([[ 12, 34 ], [ 56, 78 ]], dtype=numpy.float32)
        processed_coords = numpy.array(
            geometry.LineString(coords).coords
        )

        self.assertEqual(
            coords.tolist(),
            processed_coords.tolist()
        )

    @unittest.skipIf(not numpy, 'Numpy required')
    def test_data_destriding(self):
        coords = numpy.array([[ 12, 34 ], [ 56, 78 ]], dtype=numpy.float32)

        # Easy way to introduce striding: reverse list order
        processed_coords = numpy.array(
            geometry.LineString(coords[::-1]).coords
        )

        self.assertEqual(
            coords[::-1].tolist(),
            processed_coords.tolist()
        )


def test_coords_ctypes_deprecated():
    """
    Test that the .ctypes attribute of a CoordinateSequence raises
    a deprecation warning.
    """
    coords = geometry.LineString([[12, 34], [56, 78]]).coords
    with pytest.warns(ShapelyDeprecationWarning, match="ctypes"):
        coords.ctypes
