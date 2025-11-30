import numpy as np
import pytest

from shapely.errors import GeometryTypeError
from shapely.geometry import LinearRing, LineString, Polygon
from shapely.ops import get_segments


class TestGetSegments:
    def setup_method(self):
        self.p1 = (0, 0)
        self.p2 = (1, 1)
        self.p3 = (0, 1)
        self.line = LineString([self.p1, self.p2, self.p3])
        self.ring = LinearRing([self.p1, self.p2, self.p3, self.p1])
        self.polygon = Polygon(self.ring)

    def test_line(self):
        known = np.array(
            [LineString([self.p1, self.p2]), LineString([self.p2, self.p3])]
        )
        observed = get_segments(self.line)
        np.testing.assert_array_equal(observed, known)

    def test_ring(self):
        known = np.array(
            [
                LineString([self.p1, self.p2]),
                LineString([self.p2, self.p3]),
                LineString([self.p3, self.p1]),
            ]
        )
        observed = get_segments(self.ring)
        np.testing.assert_array_equal(observed, known)

    def test_polygon(self):
        with pytest.raises(
            GeometryTypeError, match="Getting segments from a Polygon is not supported"
        ):
            get_segments(self.polygon)
