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
    geoms = [
        Polygon([(0, 0), (1, 1), (1, 0), (0, 0)]),
        Polygon([(0, 0), (1, 1), (0, 1), (0, 0)]),
    ]
    assert shapely.coverage_is_valid(geoms)

    geoms = [
        Polygon([(0, 0), (1, 1), (1, 0), (0, 0)]),
        Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]),
    ]
    assert not shapely.coverage_is_valid(geoms)


@pytest.mark.skipif(shapely.geos_version >= (3, 12, 0), reason="requires >= 3.12")
def test_coverage_is_valid_unsupported_geos():
    geoms = [
        Polygon([(0, 0), (1, 1), (1, 0), (0, 0)]),
        Polygon([(0, 0), (1, 1), (0, 1), (0, 0)]),
    ]
    with pytest.raises(UnsupportedGEOSVersionError):
        shapely.coverage_is_valid(geoms)


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
