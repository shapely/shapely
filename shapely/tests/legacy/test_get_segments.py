import numpy as np
import pytest

from shapely.errors import GeometryTypeError
from shapely.geometry import LineString
from shapely.ops import get_segments
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

p1 = (0, 0)
p2 = (1, 0)
p3 = (1, 1)
p4 = (0, 1)

out_elbow = [LineString([p1, p2]), LineString([p2, p3])]
out_ring = [
    LineString([p1, p2]),
    LineString([p2, p3]),
    LineString([p3, p4]),
    LineString([p4, p1]),
]


@pytest.mark.parametrize(
    "geoms,expected",
    [
        [line_string, np.array(out_elbow)],
        [[line_string], np.array(out_elbow)],
        [[[line_string]], np.array(out_elbow)],
        [[[[line_string]]], np.array(out_elbow)],
        [[line_string, line_string], np.array(out_elbow + out_elbow)],
        [[[[line_string, line_string]]], np.array(out_elbow + out_elbow)],
        [[[line_string], [line_string]], np.array(out_elbow + out_elbow)],
        [line_string_z, np.array(out_elbow)],
        [line_string_m, np.array(out_elbow)],
        [line_string_zm, np.array(out_elbow)],
        [linear_ring, np.array(out_ring)],
        [[line_string, linear_ring], np.array(out_elbow + out_ring)],
    ],
)
def test_get_segments(geoms, expected):
    actual = get_segments(geoms)
    assert_geometries_equal(actual, expected)


@pytest.mark.parametrize(
    "geoms",
    [
        empty,
        point,
        multi_line_string,
        polygon,
        [polygon],
        [[polygon]],
        [[polygon], [polygon]],
        np.array([polygon]),
        np.array([empty, point, multi_line_string, polygon]),
    ],
)
def test_non_linear(geoms):
    with pytest.raises(GeometryTypeError, match=r"Check input. Getting segments from"):
        get_segments(geoms)
