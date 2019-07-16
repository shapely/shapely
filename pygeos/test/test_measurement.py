import pygeos
import pytest
import numpy as np

from .common import point_polygon_testdata
from .common import point, polygon


def test_area():
    assert pygeos.area(polygon) == 4.0


def test_area_nan():
    actual = pygeos.area(np.array([polygon, np.nan, None]))
    assert actual[0] == pygeos.area(polygon)
    assert np.isnan(actual[1])
    assert np.isnan(actual[2])


def test_distance():
    actual = pygeos.distance(*point_polygon_testdata)
    expected = [2 * 2 ** 0.5, 2 ** 0.5, 0, 0, 0, 2 ** 0.5]
    np.testing.assert_allclose(actual, expected)


def test_distance_nan():
    actual = pygeos.distance(
        np.array([point, np.nan, np.nan, point, None, None, point]),
        np.array([np.nan, point, np.nan, None, point, None, point]),
    )
    assert actual[-1] == 0.0
    assert np.isnan(actual[:-1].astype(np.float)).all()


def test_haussdorf_distance():
    # example from GEOS docs
    a = pygeos.linestrings([[0, 0], [100, 0], [10, 100], [10, 100]])
    b = pygeos.linestrings([[0, 100], [0, 10], [80, 10]])
    actual = pygeos.hausdorff_distance(a, b)
    assert actual == pytest.approx(22.360679775, abs=1e-7)


def test_haussdorf_distance_densify():
    # example from GEOS docs
    a = pygeos.linestrings([[0, 0], [100, 0], [10, 100], [10, 100]])
    b = pygeos.linestrings([[0, 100], [0, 10], [80, 10]])
    actual = pygeos.hausdorff_distance(a, b, densify=0.001)
    assert actual == pytest.approx(47.8, abs=0.1)
