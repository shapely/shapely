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
    empty_line_string,
)


@pytest.mark.skipif(shapely.geos_version < (3, 12, 0), reason="requires >= 3.12")
@pytest.mark.parametrize("geometry", all_types + all_types_z)
def test_coverage_is_valid(geometry):
    actual = shapely.coverage_is_valid([geometry])
    assert actual.ndim == 0
    assert actual.dtype == np.bool_
    assert actual.item() is True

    actual = shapely.coverage_invalid_edges([geometry])
    expected = np.array([empty_line_string], dtype=object)
    assert_geometries_equal(actual, expected)


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
    assert (shapely.coverage_invalid_edges(geoms) == empty_line_string).all()


@pytest.mark.skipif(shapely.geos_version < (3, 12, 0), reason="requires >= 3.12")
def test_coverage_is_valid_polygonal():
    # adjacent triangles
    poly1 = Polygon([(0, 0), (1, 1), (1, 0), (0, 0)])
    poly2 = Polygon([(0, 0), (1, 1), (0, 1), (0, 0)])
    assert shapely.coverage_is_valid([poly1, poly2])
    assert shapely.is_empty(shapely.coverage_invalid_edges([poly1, poly2])).all()

    # shared egde but without identical vertices
    poly2b = Polygon([(0, 0), (0.5, 0.5), (1, 1), (0, 1), (0, 0)])
    assert not shapely.coverage_is_valid([poly1, poly2b])
    result = shapely.coverage_invalid_edges([poly1, poly2b])
    expected = [LineString([(0, 0), (1, 1)]), LineString([(0, 0), (0.5, 0.5), (1, 1)])]
    assert_geometries_equal(result, expected)

    # overlap
    poly3 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    assert not shapely.coverage_is_valid([poly1, poly3])
    result = shapely.coverage_invalid_edges([poly1, poly3])
    assert not shapely.is_empty(result).any()


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

    # valid coverage -> gap_width value does not matter
    assert shapely.coverage_is_valid([poly1, poly2], gap_width=0.0)
    assert shapely.coverage_is_valid([poly1, poly2], gap_width=2.0)

    result = shapely.coverage_invalid_edges([poly1, poly2], gap_width=0.0)
    assert_geometries_equal(result, [empty_line_string] * 2)
    result = shapely.coverage_invalid_edges([poly1, poly2], gap_width=2.0)
    assert_geometries_equal(result, [empty_line_string] * 2)

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
        [empty_line_string] * 2,
    )
    assert_geometries_equal(
        shapely.coverage_invalid_edges([poly1, poly2_shift], gap_width=1.0),
        shapely.from_wkt(["LINESTRING (10 7, 10 3)", "LINESTRING (10 3, 11 5, 10 7)"]),
    )


@pytest.mark.skipif(shapely.geos_version < (3, 12, 0), reason="requires >= 3.12")
def test_coverage_invalid_edges_gufunc():
    poly1 = shapely.from_wkt("POLYGON ((0 10, 10 10, 10 7, 10 3, 10 0, 0 0, 0 10))")
    poly2 = shapely.from_wkt("POLYGON ((10 10, 20 10, 20 0, 10 0, 10 3, 10 7, 10 10))")
    poly2_extra = shapely.from_wkt(
        "POLYGON ((10 10, 20 10, 20 0, 10 0, 10 3, 10 5, 10 7, 10 10))"
    )
    poly3 = shapely.from_wkt("POLYGON ((20 10, 30 10, 30 7, 30 3, 30 0, 20 0, 20 10))")

    arr = np.array([[poly1, poly2, poly3], [poly1, poly2_extra, poly3]])
    result = shapely.lib.coverage_invalid_edges(arr, 0.0)
    expected = shapely.from_wkt(
        [
            ["LINESTRING EMPTY"] * 3,
            [
                "LINESTRING (10 7, 10 3)",
                "LINESTRING (10 3, 10 5, 10 7)",
                "LINESTRING EMPTY",
            ],
        ]
    )
    assert_geometries_equal(result, expected)

    arr2 = np.array(arr, order="F")
    result = shapely.lib.coverage_invalid_edges(arr2, 0.0)
    assert_geometries_equal(result, expected)


@pytest.mark.skipif(shapely.geos_version < (3, 12, 0), reason="GEOS < 3.12")
@pytest.mark.parametrize("geometry", all_types)
def test_coverage_simplify_scalars(geometry):
    if geometry.geom_type in {"Polygon", "MultiPolygon"}:
        actual = shapely.coverage_simplify(geometry, 0.0)
        assert isinstance(actual, Geometry)
        assert shapely.get_type_id(actual) == shapely.get_type_id(geometry)
        assert actual.equals(geometry)
    else:
        with pytest.raises(TypeError, match="incorrect geometry type"):
            shapely.coverage_simplify(geometry, 0.0)


@pytest.mark.skipif(shapely.geos_version < (3, 12, 0), reason="GEOS < 3.12")
@pytest.mark.parametrize("geometry", all_types)
def test_coverage_simplify_geom_types(geometry):
    if geometry.geom_type in {"Polygon", "MultiPolygon"}:
        actual = shapely.coverage_simplify([geometry, geometry], 0.0)
        assert isinstance(actual, np.ndarray)
        assert actual.shape == (2,)
        assert (shapely.get_type_id(actual) == shapely.get_type_id(geometry)).all()
    else:
        with pytest.raises(TypeError, match="incorrect geometry type"):
            shapely.coverage_simplify([geometry, geometry], 0.0)


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


@pytest.mark.skipif(shapely.geos_version >= (3, 14, 0), reason="requires >= 3.14")
def test_coverage_clean_unsupported_geos():
    geoms = [
        Polygon([(0, 0), (1, 1), (1, 0), (0, 0)]),
        Polygon([(0, 0), (1, 1), (0, 1), (0, 0)]),
    ]

    with pytest.raises(UnsupportedGEOSVersionError):
        shapely.coverage_clean(geoms)


@pytest.mark.skipif(shapely.geos_version < (3, 14, 0), reason="requires >= 3.14")
def test_coverage_clean_overlap():
    # overlapping polygons
    poly1 = shapely.from_wkt("POLYGON ((0 10, 10 10, 10 0, 0 0, 0 10))")
    poly2 = shapely.from_wkt("POLYGON ((9 10, 19 10, 19 0, 9 0, 9 10))")

    expected = shapely.from_wkt(
        [
            "POLYGON ((0 0, 0 10, 9 10, 10 10, 10 0, 9 0, 0 0))",
            "POLYGON ((19 10, 19 0, 10 0, 10 10, 19 10))",
        ]
    )

    result = shapely.coverage_clean([poly1, poly2])

    assert_geometries_equal(result, expected)


@pytest.mark.skipif(shapely.geos_version < (3, 14, 0), reason="requires >= 3.14")
def test_coverage_clean_gap():
    # polygons with a gap
    poly1 = shapely.from_wkt("POLYGON ((0 -1, 0 11, 10 11, 10 -1, 0 -1))")
    poly2 = shapely.from_wkt(
        "POLYGON ((10 -2, 10 0, 10.5 1, 10.5 8.5, 10 10, 10 13, 20 13, 20 -2, 10 -2))",
    )

    # gap width 0.0 (should be the same as input)
    expected_1 = shapely.from_wkt(
        [
            "POLYGON ((0 -1, 0 11, 10 11, 10 10, 10 0, 10 -1, 0 -1))",
            "POLYGON ((10 -2, 10 -1, 10 0, 10.5 1, 10.5 8.5, 10 10, "
            "10 11, 10 13, 20 13, 20 -2, 10 -2))",
        ]
    )
    result_1 = shapely.normalize(
        shapely.coverage_clean([poly1, poly2], gap_width=0),
    )
    assert_geometries_equal(result_1, expected_1)

    # gap width 1 (gap should close)
    expected_2 = shapely.from_wkt(
        [
            "POLYGON ((0 -1, 0 11, 10 11, 10 10, 10 0, 10 -1, 0 -1))",
            "POLYGON ((10 -2, 10 -1, 10 0, 10 10, 10 11, 10 13, 20 13, 20 -2, 10 -2))",
        ]
    )
    result_2 = shapely.normalize(shapely.coverage_clean([poly1, poly2], gap_width=1))
    assert_geometries_equal(result_2, expected_2)


@pytest.mark.skipif(shapely.geos_version < (3, 14, 0), reason="requires >= 3.14")
def test_coverage_clean_snapping():
    # polygons with a gap
    poly1 = shapely.from_wkt("POLYGON ((0 -1, 0 11, 10 11, 10 -1, 0 -1))")
    poly2 = shapely.from_wkt(
        "POLYGON ((10 -2, 10 0, 10.5 1, 10.5 8.5, 10 10, 10 13, 20 13, 20 -2, 10 -2))",
    )

    # snapping distance 0.0 (should be the same as input)
    expected_1 = shapely.from_wkt(
        [
            "POLYGON ((0 -1, 0 11, 10 11, 10 10, 10 0, 10 -1, 0 -1))",
            "POLYGON ((10 -2, 10 -1, 10 0, 10.5 1, 10.5 8.5, 10 10, 10 "
            "11, 10 13, 20 13, 20 -2, 10 -2))",
        ]
    )
    result_1 = shapely.normalize(
        shapely.coverage_clean([poly1, poly2], gap_width=0, snapping_distance=0),
    )
    assert_geometries_equal(result_1, expected_1)

    # snapping distance 2.0 (vertices should be snapped)
    expected_2 = shapely.from_wkt(
        [
            "POLYGON ((10.5 8.5, 10.5 1, 10 -1, 0 -1, 0 11, 10 11, 10.5 8.5))",
            "POLYGON ((10.5 1, 10.5 8.5, 10 11, 20 13, 20 -2, 10 -1, 10.5 1))",
        ]
    )
    result_2 = shapely.coverage_clean([poly1, poly2], gap_width=1, snapping_distance=2)
    assert_geometries_equal(result_2, expected_2)


@pytest.mark.skipif(shapely.geos_version < (3, 14, 0), reason="requires >= 3.14")
def test_coverage_clean_merge_strategies():
    # overlapping polygons
    poly1 = shapely.from_wkt("POLYGON ((0 0, 0 10, 10 10, 10 0, 0 0))")
    poly2 = shapely.from_wkt("POLYGON ((9 0, 9 21, 20 21, 20 0, 9 0))")
    poly3 = shapely.from_wkt("POLYGON ((6 10, 6 14, 10 14, 10 10, 6 10))")

    expected_longest_border = shapely.from_wkt(
        [
            "POLYGON ((0 0, 0 10, 6 10, 9 10, 10 10, 10 0, 9 0, 0 0))",
            "POLYGON ((9 10, 9 14, 9 21, 20 21, 20 0, 10 0, 10 10, 9 10))",
            "POLYGON ((6 10, 6 14, 9 14, 9 10, 6 10))",
        ]
    )
    result_longest_border = shapely.normalize(
        shapely.coverage_clean(
            [poly1, poly2, poly3],
            merge_strategy="longest_border",
        )
    )
    assert_geometries_equal(result_longest_border, expected_longest_border)

    expected_max_area = shapely.from_wkt(
        [
            "POLYGON ((0 0, 0 10, 6 10, 9 10, 9 0, 0 0))",
            "POLYGON ((9 0, 9 10, 9 14, 9 21, 20 21, 20 0, 10 0, 9 0))",
            "POLYGON ((6 10, 6 14, 9 14, 9 10, 6 10))",
        ]
    )
    result_max_area = shapely.normalize(
        shapely.coverage_clean(
            [poly1, poly2, poly3],
            merge_strategy="max_area",
        )
    )
    assert_geometries_equal(result_max_area, expected_max_area)

    expected_min_area = shapely.from_wkt(
        [
            "POLYGON ((0 0, 0 10, 6 10, 9 10, 10 10, 10 0, 9 0, 0 0))",
            "POLYGON ((9 14, 9 21, 20 21, 20 0, 10 0, 10 10, 10 14, 9 14))",
            "POLYGON ((6 10, 6 14, 9 14, 10 14, 10 10, 9 10, 6 10))",
        ]
    )
    result_min_area = shapely.normalize(
        shapely.coverage_clean(
            [poly1, poly2, poly3],
            merge_strategy="min_area",
        )
    )
    assert_geometries_equal(result_min_area, expected_min_area)

    expected_min_index = shapely.from_wkt(
        [
            "POLYGON ((0 0, 0 10, 6 10, 9 10, 10 10, 10 0, 9 0, 0 0))",
            "POLYGON ((9 10, 9 14, 9 21, 20 21, 20 0, 10 0, 10 10, 9 10))",
            "POLYGON ((6 10, 6 14, 9 14, 9 10, 6 10))",
        ]
    )
    result_min_index = shapely.normalize(
        shapely.coverage_clean(
            [poly1, poly2, poly3],
            merge_strategy="min_index",
        )
    )
    assert_geometries_equal(result_min_index, expected_min_index)


@pytest.mark.skipif(shapely.geos_version < (3, 14, 0), reason="requires >= 3.14")
def test_coverage_clean_invalid_merge_strategy():
    with pytest.raises(ValueError, match="'invalid' is not a valid option"):
        shapely.coverage_clean([], merge_strategy="invalid")


@pytest.mark.skipif(shapely.geos_version < (3, 14, 0), reason="requires >= 3.14")
def test_coverage_clean_scalar_parameters():
    with pytest.raises(
        ValueError,
        match="coverage_clean function called with non-scalar parameters",
    ):
        shapely.coverage_clean([], gap_width=[0, 0])
    with pytest.raises(
        ValueError,
        match="coverage_clean function called with non-scalar parameters",
    ):
        shapely.coverage_clean([], snapping_distance=[1, 1])


@pytest.mark.skipif(shapely.geos_version < (3, 14, 0), reason="requires >= 3.14")
def test_coverage_clean_overlap_multipolygons():
    # overlapping polygons
    multipoly1 = shapely.MultiPolygon(
        [
            shapely.box(0, 0, 1, 1),
            shapely.box(0.5, 0, 1.5, 1),
            shapely.box(5, 5, 6, 6),
        ]
    )

    multipoly2 = shapely.MultiPolygon(
        [
            shapely.box(0, 0.9, 1, 2),
            shapely.box(5.5, 5, 7, 6),
        ]
    )

    expected = shapely.from_wkt(
        [
            "MULTIPOLYGON (((0 0, 0 0.9, 0 1, 0.5 1, 1 1, 1.5 1, 1.5 0, 1 0, 0.5 0, "
            "0 0)), ((5 5, 5 6, 5.5 6, 6 6, 6 5, 5.5 5, 5 5)))",
            "MULTIPOLYGON (((0 1, 0 2, 1 2, 1 1, 0.5 1, 0 1)), "
            "((6 5, 6 6, 7 6, 7 5, 6 5)))",
        ],
    )
    result = shapely.normalize(shapely.coverage_clean([multipoly1, multipoly2]))

    assert_geometries_equal(result, expected)
