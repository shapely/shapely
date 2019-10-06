import pygeos
import numpy as np
import pytest

from pygeos import Geometry, GEOSException

from .common import point, all_types

CONSTRUCTIVE_NO_ARGS = (
    pygeos.boundary,
    pygeos.centroid,
    pygeos.convex_hull,
    pygeos.envelope,
    pygeos.extract_unique_points,
    pygeos.point_on_surface,
)

CONSTRUCTIVE_FLOAT_ARG = (
    pygeos.buffer,
    pygeos.delaunay_triangles,
    pygeos.simplify,
    pygeos.voronoi_polygons,
)


@pytest.mark.parametrize("geometry", all_types)
@pytest.mark.parametrize("func", CONSTRUCTIVE_NO_ARGS)
def test_no_args_array(geometry, func):
    actual = func([geometry, geometry])
    assert actual.shape == (2,)
    assert actual[0] is None or isinstance(actual[0], Geometry)


@pytest.mark.parametrize("geometry", all_types)
@pytest.mark.parametrize("func", CONSTRUCTIVE_FLOAT_ARG)
def test_float_arg_array(geometry, func):
    actual = func([geometry, geometry], 0.0)
    assert actual.shape == (2,)
    assert isinstance(actual[0], Geometry)


@pytest.mark.parametrize("geometry", all_types)
@pytest.mark.parametrize("reference", all_types)
def test_snap_array(geometry, reference):
    actual = pygeos.snap([geometry, geometry], [reference, reference], tolerance=1.0)
    assert actual.shape == (2,)
    assert isinstance(actual[0], Geometry)


@pytest.mark.parametrize("func", CONSTRUCTIVE_NO_ARGS)
def test_no_args_missing(func):
    actual = func(None)
    assert actual is None


@pytest.mark.parametrize("func", CONSTRUCTIVE_FLOAT_ARG)
def test_float_arg_missing(func):
    actual = func(None, 1.0)
    assert actual is None


@pytest.mark.parametrize("geometry", all_types)
@pytest.mark.parametrize("func", CONSTRUCTIVE_FLOAT_ARG)
def test_float_arg_nan(geometry, func):
    actual = func(geometry, float("nan"))
    assert actual is None


def test_snap_none():
    actual = pygeos.snap(None, point, tolerance=1.0)
    assert actual is None


@pytest.mark.parametrize("geometry", all_types)
def test_snap_nan_float(geometry):
    actual = pygeos.snap(geometry, point, tolerance=np.nan)
    assert actual is None
