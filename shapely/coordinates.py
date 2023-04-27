from typing import Optional

import numpy as np

import shapely
from shapely import lib, GeometryType
from shapely.lib import Geometry

__all__ = [
    "transform_interleaved",
    "transform",
    "transform_planar",
    "transform_rebuild_planar",
    "transform_rebuild",
    "count_coordinates",
    "get_coordinates",
    "set_coordinates",
]

from shapely.errors import GeometryTypeError


def transform_interleaved(geometry, transformation, include_z: bool = False):
    """Returns a copy of a geometry array with a function applied to its
    coordinates.

    With the default of ``include_z=False``, all returned geometries will be
    two-dimensional; the third dimension will be discarded, if present.
    When specifying ``include_z=True``, the returned geometries preserve
    the dimensionality of the respective input geometries.

    Parameters
    ----------
    geometry : Geometry or array_like
    transformation : function
        A function that transforms a (N, 2) or (N, 3) ndarray of float64 to
        another (N, 2) or (N, 3) ndarray of float64.
    include_z : bool, default False
        If True, include the third dimension in the coordinates array
        that is passed to the ``transformation`` function. If a
        geometry has no third dimension, the z-coordinates passed to the
        function will be NaN.

    Examples
    --------
    >>> from shapely import LineString, Point
    >>> transform_interleaved(Point(0, 0), lambda x: x + 1)
    <POINT (1 1)>
    >>> transform_interleaved(LineString([(2, 2), (4, 4)]), lambda x: x * [2, 3])
    <LINESTRING (4 6, 8 12)>
    >>> transform_interleaved(None, lambda x: x) is None
    True
    >>> transform_interleaved([Point(0, 0), None], lambda x: x).tolist()
    [<POINT (0 0)>, None]

    By default, the third dimension is ignored:

    >>> transform_interleaved(Point(0, 0, 0), lambda x: x + 1)
    <POINT (1 1)>
    >>> transform_interleaved(Point(0, 0, 0), lambda x: x + 1, include_z=True)
    <POINT Z (1 1 1)>
    """
    geometry_arr = np.array(geometry, dtype=np.object_)  # makes a copy
    coordinates = lib.get_coordinates(geometry_arr, include_z, False)
    new_coordinates = transformation(coordinates)
    # check the array to yield understandable error messages
    if not isinstance(new_coordinates, np.ndarray):
        raise ValueError("The provided transformation did not return a numpy array")
    if new_coordinates.dtype != np.float64:
        raise ValueError(
            "The provided transformation returned an array with an unexpected "
            f"dtype ({new_coordinates.dtype}, but expected {coordinates.dtype})"
        )
    if new_coordinates.shape != coordinates.shape:
        # if the shape is too small we will get a segfault
        raise ValueError(
            "The provided transformation returned an array with an unexpected "
            f"shape ({new_coordinates.shape}, but expected {coordinates.shape})"
        )
    geometry_arr = lib.set_coordinates(geometry_arr, new_coordinates)
    if geometry_arr.ndim == 0 and not isinstance(geometry, np.ndarray):
        return geometry_arr.item()
    return geometry_arr


def transform(
    geometry,
    transformation,
    include_z: Optional[bool] = False,
    interleaved: bool = True,
    rebuild: bool = False,
):
    """Returns a copy of a geometry array with a function applied to its coordinates.
    Like ``transform`` but with extra parameters.

    With the default of ``rebuild=False``, ``include_z=False``,
    all returned geometries will be two-dimensional;
    the third dimension will be discarded, if present.
    When specifying ``rebuild=False``, ``include_z=True``,
    the returned geometries preserve the dimensionality of the
    respective input geometries.
    When specifying ``rebuild=True`` the output geometries
    will get rebuilt from the transformation output

    Parameters
    ----------
    geometry : Geometry or array_like
    transformation : function
        A function that transforms coordinates. If the default value of ``interleaved=True``
        is used, the function must transform a (N, 2) or (N, 3) ndarray
        of float64 to another (N, 2) or (N, 3) ndarray of float64.
        Unless ``rebuild=True``, the function may not change N.
    include_z : bool, optional, default False
        If True, include the third dimension in the coordinates array
        that is passed to the ``transformation`` function. If a
        geometry has no third dimension, the z-coordinates passed to the
        function will be NaN.
        If False, the third dimension will never be passed into the function.
        If None, the third dimension is only passed if present.
    interleaved: bool, default True
        If set to False, the transformation function should accept 2 or 3 separate
        one-dimensional arrays (x, y and optional x).
    rebuild: bool, default False
        If set to True, the transformation function is allowed to change the number
        of coordinates per geometry and its coordiante-dimensionality.

    Examples
    --------
    >>> from shapely import LineString, Point
    >>> transform(Point(0, 0), lambda x: x + 1)
    <POINT (1 1)>
    >>> transform(LineString([(2, 2), (4, 4)]), lambda x: x * [2, 3])
    <LINESTRING (4 6, 8 12)>
    >>> transform(None, lambda x: x) is None
    True
    >>> transform([Point(0, 0), None], lambda x: x).tolist()
    [<POINT (0 0)>, None]

    By default, the third dimension is ignored:

    >>> transform(Point(0, 0, 0), lambda x: x + 1)
    <POINT (1 1)>
    >>> transform(Point(0, 0, 0), lambda x: x + 1, include_z=True)
    <POINT Z (1 1 1)>

    An identity function applicable to both types of input with interleaved=False

    >>> g1 = LineString([(1, 2), (3, 4)])

    >>> transform(g1, lambda x, y, z=None: tuple(filter(None, [x, y, z])), interleaved=False)
    <LINESTRING (1 2, 3 4)>

    example of another lambda expression:

    >>> transform(g1, lambda x, y, z=None: (x+1.0, y+1.0), interleaved=False)
    <LINESTRING (2 3, 4 5)>

    Using pyproj >= 2.1, the following example will accurately project Shapely geometries
    It transforms from EPSG:4326 (WGS84 log/lat) to EPSG:32618 (WGS84 UTM 18 North)
    Note: always_xy kwarg is required as Shapely geometries only support X,Y coordinate ordering.

    >>> try:
    ...     import pyproj
    ...     project = pyproj.Transformer.from_crs(4326, 32618, always_xy=True).transform_interleaved
    ...     p = transform(Point(-75, 50), project, interleaved=False)
    ...     assert (round(p.x), round(p.y)) == (500000, 5538631)
    ... except ImportError:
    ...     pass
    """
    if include_z is None:
        include_z = shapely.get_coordinate_dimension(geometry) == 3
    if rebuild:
        transform_wrapper = (
            transform_rebuild if interleaved else transform_rebuild_planar
        )
        vectorize = not isinstance(geometry, Geometry)
    else:
        transform_wrapper = transform_interleaved if interleaved else transform_planar
        vectorize = False
        if not np.isscalar(include_z):
            if np.all(include_z == include_z[0]):
                include_z = include_z[0]
            else:
                vectorize = True
    if vectorize:
        transform_wrapper = np.frompyfunc(transform_wrapper, 3, 1)

    return transform_wrapper(
        geometry,
        transformation,
        include_z,
    )


def transform_planar(geometry, transformation, include_z: bool = False):
    """Returns a copy of a geometry array with a function applied to its coordinates.

    Refer to `shapely.transform` (``rebuild=False``, ``interleaved=False``) for full documentation.
    """
    try:
        # First we try to apply func to x, y, z vectors.
        return transform_interleaved(
            geometry,
            lambda coords: np.array(transformation(*coords.T)).T,
            include_z=include_z,
        )
    except Exception:
        # A func that assumes x, y, z are single values will likely raise a
        # TypeError or a ValueError in which case we'll try again.
        return transform_interleaved(
            geometry,
            lambda coords: np.array([transformation(*c) for c in coords]),
            include_z=include_z,
        )


def transform_rebuild(
    geometry,
    transformation,
    include_z: bool = False,
):
    """Returns a copy of a geometry array with a function applied to its coordinates.

    Refer to `shapely.transform` (``rebuild=True``, ``interleaved=True``) for full documentation.
    """
    if geometry.is_empty:
        return geometry
    geom_type = shapely.get_type_id(geometry)
    if geom_type in [GeometryType.POINT, GeometryType.LINESTRING, GeometryType.LINEARRING, GeometryType.POLYGON]:
        return transform_rebuild_single_part(
            geometry, transformation, include_z=include_z
        )
    elif geom_type in [GeometryType.MULTIPOINT, GeometryType.MULTIPOLYGON,
                       GeometryType.MULTILINESTRING, GeometryType.GEOMETRYCOLLECTION]:
        return type(geometry)(
            [
                transform_rebuild(part, transformation, include_z=include_z)
                for part in geometry.geoms
            ]
        )
    else:
        raise GeometryTypeError(f"Type {geom_type} not recognized")


def transform_rebuild_planar(
    geometry,
    transformation,
    include_z: bool = False,
):
    """Returns a copy of a geometry array with a function applied to its coordinates.

    Refer to `shapely.transform` (``rebuild=True``, ``interleaved=False``) for full documentation.
    """
    if geometry.is_empty:
        return geometry
    geom_type = shapely.get_type_id(geometry)
    if geom_type in [GeometryType.POINT, GeometryType.LINESTRING, GeometryType.LINEARRING, GeometryType.POLYGON]:
        try:
            # First we try to apply func to x, y, z vectors.
            return transform_rebuild_single_part(
                geometry,
                lambda coords: np.array(transformation(*np.array(coords).T)).T,
                include_z=include_z,
            )
        except Exception:
            # A func that assumes x, y, z are single values will likely raise a
            # TypeError or a ValueError in which case we'll try again.
            return transform_rebuild_single_part(
                geometry,
                lambda coords: [transformation(*c) for c in coords],
                include_z=include_z,
            )
    elif geom_type in [GeometryType.MULTIPOINT, GeometryType.MULTIPOLYGON,
                       GeometryType.MULTILINESTRING, GeometryType.GEOMETRYCOLLECTION]:
        return type(geometry)(
            [
                transform_rebuild_planar(part, transformation, include_z=include_z)
                for part in geometry.geoms
            ]
        )
    else:
        raise GeometryTypeError(f"Type {geom_type} not recognized")


def transform_rebuild_single_part(geometry, transformation, include_z: bool = False):
    """Helper function for shapely.transform_rebuild, and shapely.transform_rebuild_planar
    for a single part geometries
    """
    geom_type = shapely.get_type_id(geometry)
    if geom_type in [GeometryType.POINT, GeometryType.LINESTRING, GeometryType.LINEARRING]:
        return type(geometry)(
            transformation(get_coordinates(geometry, include_z=include_z))
        )
    elif geom_type == GeometryType.POLYGON:
        shell = type(geometry.exterior)(
            transformation(get_coordinates(geometry.exterior, include_z=include_z))
        )
        holes = list(
            type(ring)(transformation(get_coordinates(ring, include_z=include_z)))
            for ring in geometry.interiors
        )
        return type(geometry)(shell, holes)


def count_coordinates(geometry):
    """Counts the number of coordinate pairs in a geometry array.

    Parameters
    ----------
    geometry : Geometry or array_like

    Examples
    --------
    >>> from shapely import LineString, Point
    >>> count_coordinates(Point(0, 0))
    1
    >>> count_coordinates(LineString([(2, 2), (4, 2)]))
    2
    >>> count_coordinates(None)
    0
    >>> count_coordinates([Point(0, 0), None])
    1
    """
    return lib.count_coordinates(np.asarray(geometry, dtype=np.object_))


def get_coordinates(geometry, include_z=False, return_index=False):
    """Gets coordinates from a geometry array as an array of floats.

    The shape of the returned array is (N, 2), with N being the number of
    coordinate pairs. With the default of ``include_z=False``, three-dimensional
    data is ignored. When specifying ``include_z=True``, the shape of the
    returned array is (N, 3).

    Parameters
    ----------
    geometry : Geometry or array_like
    include_z : bool, default False
        If, True include the third dimension in the output. If a geometry
        has no third dimension, the z-coordinates will be NaN.
    return_index : bool, default False
        If True, also return the index of each returned geometry as a separate
        ndarray of integers. For multidimensional arrays, this indexes into the
        flattened array (in C contiguous order).

    Examples
    --------
    >>> from shapely import LineString, Point
    >>> get_coordinates(Point(0, 0)).tolist()
    [[0.0, 0.0]]
    >>> get_coordinates(LineString([(2, 2), (4, 4)])).tolist()
    [[2.0, 2.0], [4.0, 4.0]]
    >>> get_coordinates(None)
    array([], shape=(0, 2), dtype=float64)

    By default the third dimension is ignored:

    >>> get_coordinates(Point(0, 0, 0)).tolist()
    [[0.0, 0.0]]
    >>> get_coordinates(Point(0, 0, 0), include_z=True).tolist()
    [[0.0, 0.0, 0.0]]

    When return_index=True, indexes are returned also:

    >>> geometries = [LineString([(2, 2), (4, 4)]), Point(0, 0)]
    >>> coordinates, index = get_coordinates(geometries, return_index=True)
    >>> coordinates.tolist(), index.tolist()
    ([[2.0, 2.0], [4.0, 4.0], [0.0, 0.0]], [0, 0, 1])
    """
    return lib.get_coordinates(
        np.asarray(geometry, dtype=np.object_), include_z, return_index
    )


def set_coordinates(geometry, coordinates):
    """Adapts the coordinates of a geometry array in-place.

    If the coordinates array has shape (N, 2), all returned geometries
    will be two-dimensional, and the third dimension will be discarded,
    if present. If the coordinates array has shape (N, 3), the returned
    geometries preserve the dimensionality of the input geometries.

    .. warning::

        The geometry array is modified in-place! If you do not want to
        modify the original array, you can do
        ``set_coordinates(arr.copy(), newcoords)``.

    Parameters
    ----------
    geometry : Geometry or array_like
    coordinates: array_like

    See Also
    --------
    transform : Returns a copy of a geometry array with a function applied to its
        coordinates.

    Examples
    --------
    >>> from shapely import LineString, Point
    >>> set_coordinates(Point(0, 0), [[1, 1]])
    <POINT (1 1)>
    >>> set_coordinates([Point(0, 0), LineString([(0, 0), (0, 0)])], [[1, 2], [3, 4], [5, 6]]).tolist()
    [<POINT (1 2)>, <LINESTRING (3 4, 5 6)>]
    >>> set_coordinates([None, Point(0, 0)], [[1, 2]]).tolist()
    [None, <POINT (1 2)>]

    Third dimension of input geometry is discarded if coordinates array does
    not include one:

    >>> set_coordinates(Point(0, 0, 0), [[1, 1]])
    <POINT (1 1)>
    >>> set_coordinates(Point(0, 0, 0), [[1, 1, 1]])
    <POINT Z (1 1 1)>
    """
    geometry_arr = np.asarray(geometry, dtype=np.object_)
    coordinates = np.atleast_2d(np.asarray(coordinates)).astype(np.float64)
    if coordinates.ndim != 2:
        raise ValueError(
            "The coordinate array should have dimension of 2 "
            f"(has {coordinates.ndim})"
        )
    n_coords = lib.count_coordinates(geometry_arr)
    if (coordinates.shape[0] != n_coords) or (coordinates.shape[1] not in {2, 3}):
        raise ValueError(
            f"The coordinate array has an invalid shape {coordinates.shape}"
        )
    lib.set_coordinates(geometry_arr, coordinates)
    if geometry_arr.ndim == 0 and not isinstance(geometry, np.ndarray):
        return geometry_arr.item()
    return geometry_arr
