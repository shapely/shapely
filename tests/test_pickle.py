import pathlib
import pickle
from pickle import dumps, loads, HIGHEST_PROTOCOL
import warnings

import shapely
from shapely.geometry import Point, LineString, LinearRing, Polygon, MultiLineString, MultiPoint, MultiPolygon, GeometryCollection, box
from shapely import wkt


import pytest


TEST_DATA = {
    "point2d": Point([(1.0, 2.0)]),
    "point3d": Point([(1.0, 2.0, 3.0)]),
    "linestring": LineString([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0)]),
    "linearring": LinearRing([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)]),
    "polygon": Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)]),
    "multipoint": MultiPoint([(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]),
    "multilinestring": MultiLineString([[(0.0, 0.0), (1.0, 1.0)], [(1.0, 2.0), (3.0, 3.0)]]),
    "multipolygon": MultiPolygon([box(0, 0, 1, 1), box(2, 2, 3, 3)]),
    "geometrycollection": GeometryCollection([Point(1.0, 2.0), box(0, 0, 1, 1)]),
    "emptypoint": wkt.loads("POINT EMPTY"),
    "emptypolygon": wkt.loads("POLYGON EMPTY"),
}
TEST_NAMES, TEST_GEOMS = zip(*TEST_DATA.items())


@pytest.mark.parametrize("geom1", TEST_GEOMS, ids=TEST_NAMES)
def test_pickle_round_trip(geom1):
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
    from shapely.testing import assert_geometries_equal

    with pytest.warns(UserWarning):
        geom = loads(SHAPELY_18_PICKLE)
    assert_geometries_equal(geom, box(0, 0, 10, 10))


HERE = pathlib.Path(__file__).parent

@pytest.mark.parametrize("fname", (HERE / "data").glob("*.pickle"), ids=lambda fname: fname.name)
def test_unpickle_pre_20(fname):
    from shapely.testing import assert_geometries_equal

    geom_type = fname.name.split("_")[0]
    expected = TEST_DATA[geom_type]
    
    with open(fname, "rb") as f:
        with pytest.warns(UserWarning):
            result = pickle.load(f)
    
    if geom_type == "emptypolygon" and "1.7.1" in fname.name:
        expected = wkt.loads("POLYGON Z EMPTY")
    assert_geometries_equal(result, expected)


if __name__ == "__main__":
    HERE = pathlib.Path(__file__).parent
    datadir = HERE / "data"
    datadir.mkdir(exist_ok=True)
    
    shapely_version = shapely.__version__
    print(shapely_version)
    print(shapely.geos.geos_version)

    for name, geom in TEST_DATA.items():
        if name == "emptypoint" and shapely.geos.geos_version < (3, 9, 0):
            # Empty Points cannot be represented in WKB
            continue
        with open(datadir / f"{name}_{shapely_version}.pickle", "wb") as f:
            pickle.dump(geom, f)
