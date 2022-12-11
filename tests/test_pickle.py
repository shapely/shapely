import pathlib
import pickle
from pickle import dumps, loads, HIGHEST_PROTOCOL
import warnings

import shapely
from shapely.geometry import Point, LineString, LinearRing, Polygon, MultiLineString, MultiPoint, MultiPolygon, GeometryCollection, box
from shapely import wkt

import pytest


HERE = pathlib.Path(__file__).parent


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


@pytest.mark.parametrize("fname", (HERE / "data").glob("*.pickle"), ids=lambda fname: fname.name)
def test_unpickle_pre_20(fname):
    from shapely.testing import assert_geometries_equal

    geom_type = fname.name.split("_")[0]
    expected = TEST_DATA[geom_type]

    with open(fname, "rb") as f:
        with pytest.warns(UserWarning):
            result = pickle.load(f)

    assert_geometries_equal(result, expected)


if __name__ == "__main__":
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
