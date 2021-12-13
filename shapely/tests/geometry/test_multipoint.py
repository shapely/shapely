from .test_multi import MultiGeometryTestCase
import numpy as np

import pytest

from shapely.errors import EmptyPartError
from shapely.geometry import Point, MultiPoint
from shapely.geometry.base import dump_coords


class TestMultiPoint(MultiGeometryTestCase):

    def test_multipoint(self):

        # From coordinate tuples
        geom = MultiPoint(((1.0, 2.0), (3.0, 4.0)))
        assert len(geom.geoms) == 2
        assert dump_coords(geom) == [[(1.0, 2.0)], [(3.0, 4.0)]]

        # From points
        geom = MultiPoint((Point(1.0, 2.0), Point(3.0, 4.0)))
        assert len(geom.geoms) == 2
        assert dump_coords(geom) == [[(1.0, 2.0)], [(3.0, 4.0)]]

        # From another multi-point
        geom2 = MultiPoint(geom)
        assert len(geom2.geoms) == 2
        assert dump_coords(geom2) == [[(1.0, 2.0)], [(3.0, 4.0)]]

        # Sub-geometry Access
        assert isinstance(geom.geoms[0], Point)
        assert geom.geoms[0].x == 1.0
        assert geom.geoms[0].y == 2.0
        with pytest.raises(IndexError):  # index out of range
            geom.geoms[2]

        # Geo interface
        assert geom.__geo_interface__ == {'type': 'MultiPoint',
                          'coordinates': ((1.0, 2.0), (3.0, 4.0))}

    def test_multipoint_from_numpy(self):
        # Construct from a numpy array
        geom = MultiPoint(np.array([[0.0, 0.0], [1.0, 2.0]]))
        assert isinstance(geom, MultiPoint)
        assert len(geom.geoms) == 2
        assert dump_coords(geom) == [[(0.0, 0.0)], [(1.0, 2.0)]]

    def test_subgeom_access(self):
        p0 = Point(1.0, 2.0)
        p1 = Point(3.0, 4.0)
        self.subgeom_access_test(MultiPoint, [p0, p1])

    def test_create_multi_with_empty_component(self):
        with pytest.raises(EmptyPartError, match="Can't create MultiPoint with empty component"):
            wkt = MultiPoint([Point(0, 0), Point()]).wkt



def test_multipoint_array_coercion():
    geom = MultiPoint(((1.0, 2.0), (3.0, 4.0)))
    arr = np.array(geom)
    assert arr.ndim == 0
    assert arr.size == 1
    assert arr.dtype == np.dtype("object")
    assert arr.item() == geom


@pytest.mark.filterwarnings("error:An exception was ignored")  # NumPy 1.21
def test_numpy_object_array():
    geom = MultiPoint(((1.0, 2.0), (3.0, 4.0)))
    ar = np.empty(1, object)
    ar[:] = [geom]
    assert ar[0] == geom


def test_len_raises():
    geom = MultiPoint([[5.0, 6.0], [7.0, 8.0]])
    with pytest.raises(TypeError):
        len(geom)
