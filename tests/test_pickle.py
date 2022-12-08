import pytest
from shapely.geometry import Point, LineString, LinearRing, Polygon, MultiPoint, box
from pickle import dumps, loads, HIGHEST_PROTOCOL
from shapely.testing import assert_geometries_equal
import warnings

TEST_DATA = {
    "point2d": (Point, [(1.0, 2.0)]),
    "point3d": (Point, [(1.0, 2.0, 3.0)]),
    "linestring": (LineString, [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0)]),
    "linearring": (LinearRing, [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)]),
    "polygon": (Polygon, [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)]),
    "multipoint": (MultiPoint, [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]),
}
TEST_NAMES, TEST_DATA = zip(*TEST_DATA.items())
@pytest.mark.parametrize("cls,coords", TEST_DATA, ids=TEST_NAMES)
def test_pickle_round_trip(cls, coords):
    geom1 = cls(coords)
    assert geom1.has_z == (len(coords[0]) == 3)
    data = dumps(geom1, HIGHEST_PROTOCOL)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        geom2 = loads(data)
    assert geom2.has_z == geom1.has_z
    assert type(geom2) is type(geom1)
    assert geom2.geom_type == geom1.geom_type
    assert geom2.wkt == geom1.wkt


SHAPELY_18_PICKLE = (
    b'\x80\x04\x95\x8c\x00\x00\x00\x00\x00\x00\x00\x8c\x18shapely.geometry.polygon'
    b'\x94\x8c\x07Polygon\x94\x93\x94)R\x94C]\x01\x03\x00\x00\x00\x01\x00\x00\x00'
    b'\x05\x00\x00\x00\x00\x00\x00\x00\x00\x00$@\x00\x00\x00\x00\x00\x00\x00\x00'
    b'\x00\x00\x00\x00\x00\x00$@\x00\x00\x00\x00\x00\x00$@\x00\x00\x00\x00\x00\x00'
    b'\x00\x00\x00\x00\x00\x00\x00\x00$@\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
    b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00$@\x00\x00\x00\x00\x00\x00\x00\x00'
    b'\x94b.'
)


def test_unpickle_shapely_18():
    with pytest.warns(UserWarning):
        geom = loads(SHAPELY_18_PICKLE)
    assert_geometries_equal(geom, box(0, 0, 10, 10))
