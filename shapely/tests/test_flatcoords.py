import numpy as np
import pytest
from numpy.testing import assert_allclose

import shapely
from shapely.testing import assert_geometries_equal

from .common import (
    geometry_collection,
    line_string,
    line_string_z,
    linear_ring,
    multi_line_string,
    multi_line_string_z,
    multi_point,
    multi_point_z,
    multi_polygon,
    multi_polygon_z,
    point,
    point_z,
    polygon,
    polygon_z,
)

all_types = (
    point,
    line_string,
    polygon,
    multi_point,
    multi_line_string,
    multi_polygon,
)

all_types_3d = (
    point_z,
    line_string_z,
    polygon_z,
    multi_point_z,
    multi_line_string_z,
    multi_polygon_z,
)

all_types_not_supported = (
    linear_ring,
    geometry_collection,
)


@pytest.mark.parametrize("geom", all_types)
def test_roundtrip(geom):
    actual = shapely.from_ragged_array(*shapely.to_ragged_array([geom, geom]))
    assert_geometries_equal(actual, [geom, geom])


@pytest.mark.parametrize("geom", all_types_3d)
def test_roundtrip_3d(geom):
    actual = shapely.from_ragged_array(
        *shapely.to_ragged_array([geom, geom], include_z=True)
    )
    assert_geometries_equal(actual, [geom, geom])


@pytest.mark.parametrize("geom", all_types)
def test_include_z(geom):
    _, coords, _ = shapely.to_ragged_array([geom, geom], include_z=True)
    # For 2D geoms, z coords are filled in with NaN
    assert np.isnan(coords[:, 2]).all()


@pytest.mark.parametrize("geom", all_types_3d)
def test_include_z_false(geom):
    _, coords, _ = shapely.to_ragged_array([geom, geom], include_z=False)
    # For 3D geoms, z coords are dropped
    assert coords.shape[1] == 2


@pytest.mark.parametrize("geom", all_types_not_supported)
def test_raise_geometry_type(geom):
    with pytest.raises(ValueError):
        shapely.to_ragged_array([geom, geom])


def test_points():
    arr = shapely.from_wkt(
        [
            "POINT (0 0)",
            "POINT (1 1)",
            "POINT EMPTY",
            "POINT EMPTY",
            "POINT (4 4)",
            "POINT EMPTY",
        ]
    )
    typ, result, offsets = shapely.to_ragged_array(arr)
    expected = np.array(
        [[0, 0], [1, 1], [np.nan, np.nan], [np.nan, np.nan], [4, 4], [np.nan, np.nan]]
    )
    assert typ == shapely.GeometryType.POINT
    assert_allclose(result, expected)
    assert len(offsets) == 0

    geoms = shapely.from_ragged_array(typ, result)
    assert_geometries_equal(geoms, arr)


def test_linestrings():
    arr = shapely.from_wkt(
        [
            "LINESTRING (30 10, 10 30, 40 40)",
            "LINESTRING (40 40, 30 30, 40 20, 30 10)",
            "LINESTRING EMPTY",
            "LINESTRING EMPTY",
            "LINESTRING (10 10, 20 20, 10 40)",
            "LINESTRING EMPTY",
        ]
    )
    typ, coords, offsets = shapely.to_ragged_array(arr)
    expected = np.array(
        [
            [30.0, 10.0],
            [10.0, 30.0],
            [40.0, 40.0],
            [40.0, 40.0],
            [30.0, 30.0],
            [40.0, 20.0],
            [30.0, 10.0],
            [10.0, 10.0],
            [20.0, 20.0],
            [10.0, 40.0],
        ]
    )
    expected_offsets = np.array([0, 3, 7, 7, 7, 10, 10])
    assert typ == shapely.GeometryType.LINESTRING
    assert_allclose(coords, expected)
    assert len(offsets) == 1
    assert_allclose(offsets[0], expected_offsets)

    result = shapely.from_ragged_array(typ, coords, offsets)
    assert_geometries_equal(result, arr)


def test_polygons():
    arr = shapely.from_wkt(
        [
            "POLYGON ((30 10, 40 40, 20 40, 10 20, 30 10))",
            "POLYGON ((35 10, 45 45, 15 40, 10 20, 35 10), (20 30, 35 35, 30 20, 20 30))",
            "POLYGON EMPTY",
            "POLYGON EMPTY",
            "POLYGON ((30 10, 40 40, 20 40, 10 20, 30 10))",
            "POLYGON EMPTY",
        ]
    )
    typ, coords, offsets = shapely.to_ragged_array(arr)
    expected = np.array(
        [
            [30.0, 10.0],
            [40.0, 40.0],
            [20.0, 40.0],
            [10.0, 20.0],
            [30.0, 10.0],
            [35.0, 10.0],
            [45.0, 45.0],
            [15.0, 40.0],
            [10.0, 20.0],
            [35.0, 10.0],
            [20.0, 30.0],
            [35.0, 35.0],
            [30.0, 20.0],
            [20.0, 30.0],
            [30.0, 10.0],
            [40.0, 40.0],
            [20.0, 40.0],
            [10.0, 20.0],
            [30.0, 10.0],
        ]
    )
    expected_offsets1 = np.array([0, 5, 10, 14, 19])
    expected_offsets2 = np.array([0, 1, 3, 3, 3, 4, 4])

    assert typ == shapely.GeometryType.POLYGON
    assert_allclose(coords, expected)
    assert len(offsets) == 2
    assert_allclose(offsets[0], expected_offsets1)
    assert_allclose(offsets[1], expected_offsets2)

    result = shapely.from_ragged_array(typ, coords, offsets)
    assert_geometries_equal(result, arr)


def test_multipoints():
    arr = shapely.from_wkt(
        [
            "MULTIPOINT (10 40, 40 30, 20 20, 30 10)",
            "MULTIPOINT (30 10)",
            "MULTIPOINT EMPTY",
            "MULTIPOINT EMPTY",
            "MULTIPOINT (30 10, 10 30, 40 40)",
            "MULTIPOINT EMPTY",
        ]
    )
    typ, coords, offsets = shapely.to_ragged_array(arr)
    expected = np.array(
        [
            [10.0, 40.0],
            [40.0, 30.0],
            [20.0, 20.0],
            [30.0, 10.0],
            [30.0, 10.0],
            [30.0, 10.0],
            [10.0, 30.0],
            [40.0, 40.0],
        ]
    )
    expected_offsets = np.array([0, 4, 5, 5, 5, 8, 8])

    assert typ == shapely.GeometryType.MULTIPOINT
    assert_allclose(coords, expected)
    assert len(offsets) == 1
    assert_allclose(offsets[0], expected_offsets)

    result = shapely.from_ragged_array(typ, coords, offsets)
    assert_geometries_equal(result, arr)


def test_multilinestrings():
    arr = shapely.from_wkt(
        [
            "MULTILINESTRING ((30 10, 10 30, 40 40))",
            "MULTILINESTRING ((10 10, 20 20, 10 40),(40 40, 30 30, 40 20, 30 10))",
            "MULTILINESTRING EMPTY",
            "MULTILINESTRING EMPTY",
            "MULTILINESTRING ((35 10, 45 45), (15 40, 10 20), (30 10, 10 30, 40 40))",
            "MULTILINESTRING EMPTY",
        ]
    )
    typ, coords, offsets = shapely.to_ragged_array(arr)
    expected = np.array(
        [
            [30.0, 10.0],
            [10.0, 30.0],
            [40.0, 40.0],
            [10.0, 10.0],
            [20.0, 20.0],
            [10.0, 40.0],
            [40.0, 40.0],
            [30.0, 30.0],
            [40.0, 20.0],
            [30.0, 10.0],
            [35.0, 10.0],
            [45.0, 45.0],
            [15.0, 40.0],
            [10.0, 20.0],
            [30.0, 10.0],
            [10.0, 30.0],
            [40.0, 40.0],
        ]
    )
    expected_offsets1 = np.array([0, 3, 6, 10, 12, 14, 17])
    expected_offsets2 = np.array([0, 1, 3, 3, 3, 6, 6])

    assert typ == shapely.GeometryType.MULTILINESTRING
    assert_allclose(coords, expected)
    assert len(offsets) == 2
    assert_allclose(offsets[0], expected_offsets1)
    assert_allclose(offsets[1], expected_offsets2)

    result = shapely.from_ragged_array(typ, coords, offsets)
    assert_geometries_equal(result, arr)


def test_multipolygons():
    arr = shapely.from_wkt(
        [
            "MULTIPOLYGON (((35 10, 45 45, 15 40, 10 20, 35 10), (20 30, 35 35, 30 20, 20 30)))",
            "MULTIPOLYGON (((40 40, 20 45, 45 30, 40 40)),((20 35, 10 30, 10 10, 30 5, 45 20, 20 35),(30 20, 20 15, 20 25, 30 20)))",
            "MULTIPOLYGON EMPTY",
            "MULTIPOLYGON EMPTY",
            "MULTIPOLYGON (((40 40, 20 45, 45 30, 40 40)))",
            "MULTIPOLYGON EMPTY",
        ]
    )
    typ, coords, offsets = shapely.to_ragged_array(arr)
    expected = np.array(
        [
            [35.0, 10.0],
            [45.0, 45.0],
            [15.0, 40.0],
            [10.0, 20.0],
            [35.0, 10.0],
            [20.0, 30.0],
            [35.0, 35.0],
            [30.0, 20.0],
            [20.0, 30.0],
            [40.0, 40.0],
            [20.0, 45.0],
            [45.0, 30.0],
            [40.0, 40.0],
            [20.0, 35.0],
            [10.0, 30.0],
            [10.0, 10.0],
            [30.0, 5.0],
            [45.0, 20.0],
            [20.0, 35.0],
            [30.0, 20.0],
            [20.0, 15.0],
            [20.0, 25.0],
            [30.0, 20.0],
            [40.0, 40.0],
            [20.0, 45.0],
            [45.0, 30.0],
            [40.0, 40.0],
        ]
    )
    expected_offsets1 = np.array([0, 5, 9, 13, 19, 23, 27])
    expected_offsets2 = np.array([0, 2, 3, 5, 6])
    expected_offsets3 = np.array([0, 1, 3, 3, 3, 4, 4])

    assert typ == shapely.GeometryType.MULTIPOLYGON
    assert_allclose(coords, expected)
    assert len(offsets) == 3
    assert_allclose(offsets[0], expected_offsets1)
    assert_allclose(offsets[1], expected_offsets2)
    assert_allclose(offsets[2], expected_offsets3)

    result = shapely.from_ragged_array(typ, coords, offsets)
    assert_geometries_equal(result, arr)
