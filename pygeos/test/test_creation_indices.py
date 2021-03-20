import pygeos
import pytest
import numpy as np
from .common import point, line_string, linear_ring, polygon, empty

geom_coll = pygeos.geometrycollections


@pytest.mark.parametrize(
    "geometries",
    [
        np.array([1, 2], dtype=np.int32),
        None,
        np.array([[point]]),
        "hello",
    ],
)
def test_invalid_geometries(geometries):
    with pytest.raises(TypeError):
        pygeos.geometrycollections(geometries, indices=[0, 1])


@pytest.mark.parametrize(
    "indices",
    [
        np.array([point]),
        " hello",
        [0, 1],  # wrong length
    ],
)
def test_invalid_indices(indices):
    with pytest.raises((TypeError, ValueError)):
        pygeos.geometrycollections([point], indices=indices)


@pytest.mark.parametrize(
    "geometries,indices,expected",
    [
        ([point, line_string], [0, 0], [geom_coll([point, line_string])]),
        ([point, line_string], [0, 1], [geom_coll([point]), geom_coll([line_string])]),
        (
            [point, line_string],
            [1, 1],
            [geom_coll([]), geom_coll([point, line_string])],
        ),
        ([point, None], [0, 0], [geom_coll([point])]),
        ([point, None], [0, 1], [geom_coll([point]), geom_coll([])]),
        ([point, None, line_string], [0, 0, 0], [geom_coll([point, line_string])]),
    ],
)
def test_geometrycollections(geometries, indices, expected):
    actual = pygeos.geometrycollections(geometries, indices=indices)
    assert pygeos.equals(actual, expected).all()


def test_multipoints():
    actual = pygeos.multipoints(
        [point],
        indices=[0],
    )
    assert pygeos.equals(actual, pygeos.multipoints([point])).all()


def test_multilinestrings():
    actual = pygeos.multilinestrings([line_string], indices=[0])
    assert pygeos.equals(actual, pygeos.multilinestrings([line_string])).all()


def test_multilinearrings():
    actual = pygeos.multilinestrings(
        [linear_ring],
        indices=[0],
    )
    assert pygeos.equals(actual, pygeos.multilinestrings([linear_ring])).all()


def test_multipolygons():
    actual = pygeos.multipolygons(
        [polygon],
        indices=[0],
    )
    assert pygeos.equals(actual, pygeos.multipolygons([polygon])).all()


@pytest.mark.parametrize(
    "geometries,func",
    [
        ([line_string], pygeos.multipoints),
        ([polygon], pygeos.multipoints),
        ([point], pygeos.multilinestrings),
        ([polygon], pygeos.multilinestrings),
        ([point], pygeos.multipolygons),
        ([line_string], pygeos.multipolygons),
    ],
)
def test_incompatible_types(geometries, func):
    with pytest.raises(TypeError):
        func(geometries, indices=[0])
