import pygeos
import numpy as np

from .common import point


def test_intersection():
    poly1, poly2 = pygeos.box(0, 0, 10, 10), pygeos.box(5, 5, 20, 20)
    actual = pygeos.intersection(poly1, poly2)
    expected = pygeos.box(5, 5, 10, 10)
    assert pygeos.equals(actual, expected)


def test_union():
    poly1, poly2 = pygeos.box(0, 0, 10, 10), pygeos.box(10, 0, 20, 10)
    actual = pygeos.union(poly1, poly2)
    expected = pygeos.box(0, 0, 20, 10)
    assert pygeos.equals(actual, expected)


def test_intersection_nan():
    actual = pygeos.intersection(
        np.array([point, np.nan, np.nan, point, None, None, point]),
        np.array([np.nan, point, np.nan, None, point, None, point]),
    )
    assert pygeos.equals(actual[-1], point)
    assert np.isnan(actual[:-1].astype(np.float)).all()
