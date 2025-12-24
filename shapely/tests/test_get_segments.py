import numpy as np
import pytest

from shapely import get_segments
from shapely.geometry import LineString
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
    empty,
    line_string,
    line_string_m,
    line_string_z,
    line_string_zm,
    linear_ring,
    multi_line_string,
    point,
    polygon,
)

p1 = [0, 0]
p2 = [1, 0]
p3 = [1, 1]
p4 = [0, 1]

z = 4
p1z = [0, 0, z]
p2z = [1, 0, z]
p3z = [1, 1, z]
p4z = [0, 1, z]

p1nan = [0, 0, np.nan]
p2nan = [1, 0, np.nan]
p3nan = [1, 1, np.nan]
p4nan = [0, 1, np.nan]


out_elbow = [LineString([p1, p2]), LineString([p2, p3])]
out_elbow_z = [LineString([p1z, p2z]), LineString([p2z, p3z])]
out_elbow_nan = [LineString([p1nan, p2nan]), LineString([p2nan, p3nan])]

out_ring = [
    LineString([p1, p2]),
    LineString([p2, p3]),
    LineString([p3, p4]),
    LineString([p4, p1]),
]
out_ring_z = [
    LineString([p1z, p2z]),
    LineString([p2z, p3z]),
    LineString([p3z, p4z]),
    LineString([p4z, p1z]),
]
out_ring_nan = [
    LineString([p1nan, p2nan]),
    LineString([p2nan, p3nan]),
    LineString([p3nan, p4nan]),
    LineString([p4nan, p1nan]),
]


@pytest.mark.parametrize(
    "geoms,expected",
    [
        [line_string, np.array(out_elbow)],
        [[line_string], np.array(out_elbow)],
        [[line_string, line_string], np.array(out_elbow + out_elbow)],
        [line_string_z, np.array(out_elbow)],
        [line_string_m, np.array(out_elbow)],
        [line_string_zm, np.array(out_elbow)],
        [linear_ring, np.array(out_ring)],
        [[line_string, linear_ring], np.array(out_elbow + out_ring)],
    ],
)
@pytest.mark.parametrize(
    "create_style", ["loop", "list-comprehension", "map", "indices"]
)
def test_get_segments_defaults(geoms, expected, create_style):
    actual = get_segments(geoms, create_style=create_style)
    assert_geometries_equal(actual, expected)


@pytest.mark.parametrize(
    "geoms",
    [
        [[line_string, line_string]],
        np.array([[line_string], [line_string]]),
    ],
)
def test_non_1d_array(geoms):
    with pytest.raises(ValueError, match=r"Array should be one dimensional"):
        get_segments(geoms)


@pytest.mark.parametrize(
    "geoms",
    [
        [[line_string], line_string],
        [line_string, [line_string]],
    ],
)
def test_ragged_array(geoms):
    with pytest.raises(TypeError, match=r"One of the arguments is of incorrect type"):
        get_segments(geoms)


@pytest.mark.parametrize(
    "geoms",
    [
        [empty, line_string],
        point,
        multi_line_string,
        polygon,
        [polygon],
        [polygon, polygon],
        np.array([polygon]),
        np.array([empty, point, multi_line_string, polygon]),
    ],
)
def test_non_linear(geoms):
    with pytest.raises(ValueError, match=r"Geometry type is not supported"):
        get_segments(geoms)
