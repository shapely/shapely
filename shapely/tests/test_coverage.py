import numpy as np
import pytest

import shapely
from shapely import (
    Geometry,
    LineString,
    MultiPolygon,
    Polygon,
)
from shapely.errors import UnsupportedGEOSVersionError
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
    all_types,
    all_types_z,
)


@pytest.mark.skipif(shapely.geos_version < (3, 12, 0), reason="requires >= 3.12")
@pytest.mark.parametrize("geometry", all_types + all_types_z)
def test_coverage_is_valid(geometry):
    actual = shapely.coverage_is_valid([geometry])
    assert actual.ndim == 0
    assert actual.dtype == np.bool_
    assert actual.item() is True


@pytest.mark.skipif(shapely.geos_version < (3, 12, 0), reason="requires >= 3.12")
def test_coverage_is_valid_non_polygonal():
    # non-polygonal geometries are ignored to validate the coverage
    # (e.g. even if you have crossing linestrings)
    geoms = [
        LineString([(0, 0), (1, 1)]),
        LineString([(1, 0), (0, 1)]),
        Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)]),
    ]
    assert shapely.coverage_is_valid(geoms)


@pytest.mark.skipif(shapely.geos_version < (3, 12, 0), reason="requires >= 3.12")
def test_coverage_is_valid_polygonal():
    # adjacent triangles
    poly1 = Polygon([(0, 0), (1, 1), (1, 0), (0, 0)])
    poly2 = Polygon([(0, 0), (1, 1), (0, 1), (0, 0)])
    assert shapely.coverage_is_valid([poly1, poly2])

    # shared egde but without identical vertices
    poly2b = Polygon([(0, 0), (0.5, 0.5), (1, 1), (0, 1), (0, 0)])
    assert not shapely.coverage_is_valid([poly1, poly2b])

    # overlap
    poly3 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    assert not shapely.coverage_is_valid([poly1, poly3])


@pytest.mark.skipif(shapely.geos_version < (3, 12, 0), reason="requires >= 3.12")
def test_coverage_is_valid_gap_width():
    # shared edge of boxes with multiple vertices
    poly1 = shapely.from_wkt("POLYGON ((0 10, 10 10, 10 7, 10 3, 10 0, 0 0, 0 10))")
    poly2 = shapely.from_wkt("POLYGON ((10 10, 20 10, 20 0, 10 0, 10 3, 10 7, 10 10))")

    # extra vertex in middle of shared edge
    poly2_extra = shapely.from_wkt(
        "POLYGON ((10 10, 20 10, 20 0, 10 0, 10 3, 10 5, 10 7, 10 10))"
    )
    # extra vertex shifted -> gap of max 1 wide
    poly2_shift = shapely.from_wkt(
        "POLYGON ((10 10, 20 10, 20 0, 10 0, 10 3, 11 5, 10 7, 10 10))"
    )
    # poly3 = shapely.from_wkt("POLYGON ((20 10, 30 10, 30 0, 20 0, 20 10))")

    # valid coverage -> gap_width value does not matter
    assert shapely.coverage_is_valid([poly1, poly2], gap_width=0.0)
    assert shapely.coverage_is_valid([poly1, poly2], gap_width=2.0)

    expected_valid = shapely.from_wkt(["LINESTRING EMPTY"] * 2)
    result = shapely.coverage_invalid_edges([poly1, poly2], gap_width=0.0)
    assert_geometries_equal(result, expected_valid)
    result = shapely.coverage_invalid_edges([poly1, poly2], gap_width=2.0)
    assert_geometries_equal(result, expected_valid)

    # invalid coverage -> gap_width value does not matter
    assert not shapely.coverage_is_valid([poly1, poly2_extra], gap_width=0.0)
    assert not shapely.coverage_is_valid([poly1, poly2_extra], gap_width=2.0)

    expected = shapely.from_wkt(
        ["LINESTRING (10 7, 10 3)", "LINESTRING (10 3, 10 5, 10 7)"]
    )
    result = shapely.coverage_invalid_edges([poly1, poly2_extra], gap_width=0.0)
    assert_geometries_equal(result, expected)
    result = shapely.coverage_invalid_edges([poly1, poly2_extra], gap_width=2.0)
    assert_geometries_equal(result, expected)

    # coverage with gap of 1 unit wide
    assert shapely.coverage_is_valid([poly1, poly2_shift], gap_width=0.0)
    assert shapely.coverage_is_valid([poly1, poly2_shift], gap_width=0.5)
    assert not shapely.coverage_is_valid([poly1, poly2_shift], gap_width=1.0)
    assert not shapely.coverage_is_valid([poly1, poly2_shift], gap_width=1.5)
    # TODO why this behaviour?
    assert shapely.coverage_is_valid([poly1, poly2_shift], gap_width=2.0)

    assert_geometries_equal(
        shapely.coverage_invalid_edges([poly1, poly2_shift], gap_width=0.0),
        expected_valid,
    )
    expected = shapely.from_wkt(
        ["LINESTRING (10 7, 10 3)", "LINESTRING (10 3, 11 5, 10 7)"]
    )
    result = shapely.coverage_invalid_edges([poly1, poly2_shift], gap_width=1.0)
    assert_geometries_equal(result, expected)


@pytest.mark.skipif(shapely.geos_version < (3, 12, 0), reason="GEOS < 3.12")
@pytest.mark.parametrize("geometry", all_types)
def test_coverage_simplify_scalars(geometry):
    actual = shapely.coverage_simplify(geometry, 0.0)
    assert isinstance(actual, Geometry)
    assert shapely.get_type_id(actual) == shapely.get_type_id(geometry)
    # Anything other than MultiPolygon or a GeometryCollection is returned as-is
    if shapely.get_type_id(geometry) not in (3, 6):
        assert actual.equals(geometry)


@pytest.mark.skipif(shapely.geos_version < (3, 12, 0), reason="GEOS < 3.12")
@pytest.mark.parametrize("geometry", all_types)
def test_coverage_simplify_geom_types(geometry):
    actual = shapely.coverage_simplify([geometry, geometry], 0.0)
    assert isinstance(actual, np.ndarray)
    assert actual.shape == (2,)
    assert (shapely.get_type_id(actual) == shapely.get_type_id(geometry)).all()


@pytest.mark.skipif(shapely.geos_version < (3, 12, 0), reason="GEOS < 3.12")
def test_coverage_simplify_multipolygon():
    mp = MultiPolygon(
        [
            Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)]),
            Polygon([(2, 2), (2, 3), (3, 3), (3, 2), (2, 2)]),
        ]
    )
    actual = shapely.coverage_simplify(mp, 1)
    assert actual.equals(
        shapely.from_wkt(
            "MULTIPOLYGON (((0 1, 1 1, 1 0, 0 1)), ((2 3, 3 3, 3 2, 2 3)))"
        )
    )


@pytest.mark.skipif(shapely.geos_version < (3, 12, 0), reason="GEOS < 3.12")
def test_coverage_simplify_array():
    polygons = np.array(
        [
            shapely.Polygon([(0, 0), (20, 0), (20, 10), (10, 5), (0, 10), (0, 0)]),
            shapely.Polygon([(0, 10), (10, 5), (20, 10), (20, 20), (0, 20), (0, 10)]),
        ]
    )
    low_tolerance = shapely.coverage_simplify(polygons, 1)
    mid_tolerance = shapely.coverage_simplify(polygons, 8)
    high_tolerance = shapely.coverage_simplify(polygons, 10)

    assert shapely.equals(low_tolerance, shapely.normalize(polygons)).all()
    assert shapely.equals(
        mid_tolerance,
        shapely.from_wkt(
            [
                "POLYGON ((20 10, 0 10, 0 0, 20 0, 20 10))",
                "POLYGON ((20 10, 0 10, 0 20, 20 20, 20 10))",
            ]
        ),
    ).all()
    assert shapely.equals(
        high_tolerance,
        shapely.from_wkt(
            [
                "POLYGON ((20 10, 0 10, 20 0, 20 10))",
                "POLYGON ((20 10, 0 10, 0 20, 20 10))",
            ]
        ),
    ).all()

    no_boundary = shapely.coverage_simplify(polygons, 10, simplify_boundary=False)
    assert shapely.equals(
        no_boundary,
        shapely.from_wkt(
            [
                "POLYGON ((20 10, 0 10, 0 0, 20 0, 20 10))",
                "POLYGON ((20 10, 0 10, 0 20, 20 20, 20 10))",
            ]
        ),
    ).all()


@pytest.mark.skipif(shapely.geos_version >= (3, 12, 0), reason="requires >= 3.12")
def test_coverage_unsupported_geos():
    geoms = [
        Polygon([(0, 0), (1, 1), (1, 0), (0, 0)]),
        Polygon([(0, 0), (1, 1), (0, 1), (0, 0)]),
    ]
    with pytest.raises(UnsupportedGEOSVersionError):
        shapely.coverage_is_valid(geoms)

    with pytest.raises(UnsupportedGEOSVersionError):
        shapely.coverage_invalid_edges(geoms)

    with pytest.raises(UnsupportedGEOSVersionError):
        shapely.coverage_simplify(geoms, 1.0)
