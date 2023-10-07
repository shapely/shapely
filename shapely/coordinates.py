from typing import Optional

import numpy as np

import shapely
from shapely import GeometryType, lib

__all__ = [
    "transform",
    "count_coordinates",
    "get_coordinates",
    "set_coordinates",
    "transform_resize",
]


def transform(
    geometry,
    transformation,
    include_z: Optional[bool] = False,
    interleaved: bool = True,
):
    """Returns a copy of a geometry array with a function applied to its
    coordinates.

    With the default of ``include_z=False``, all returned geometries will be
    two-dimensional; the third dimension will be discarded, if present.
    When specifying ``include_z=True``, the returned geometries preserve
    the dimensionality of the respective input geometries.

    This function differs with `transform_resize` in the following ways:

    - It also accepts arrays of Geometry objects.
    - The parameters to the transformation function are numpy arrays.
    - The number of coordinates per Geometry is not allowed to change.

    Parameters
    ----------
    geometry : Geometry or array_like
    transformation : function
        A function that transforms a (N, 2) or (N, 3) ndarray of float64 to
        another (N, 2) or (N, 3) ndarray of float64.
        The function may not change N.
    include_z : bool, optional, default False
        If False, always return 2D geometries.
        If True, the data being passed to the
        transformation function will include the third dimension
        (if a geometry has no third dimension, the z-coordinates
        will be NaN). If None, will infer the dimensionality per
        input geometry using ``has_z``, which may result in 2 calls to
        the transformation function. Note that this inference
        can be unreliable with empty geometries or NaN coordinates: for a
        guaranteed result, it is recommended to specify ``include_z`` explicitly.
    interleaved : bool, default True
        If set to False, the transformation function should accept 2 or 3 separate
        one-dimensional arrays (x, y and optional z) instead of a single
        two-dimensional array.

    See Also
    --------
    has_z : Returns a copy of a geometry array with a function applied to its
        coordinates.
    transform_resize : Transform single Geometry objects, optionally resizing them.

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

    The presence of a third dimension can be automatically detected, or
    controlled explicitly:

    >>> transform(Point(0, 0, 0), lambda x: x + 1)
    <POINT (1 1)>
    >>> transform(Point(0, 0, 0), lambda x: x + 1, include_z=True)
    <POINT Z (1 1 1)>
    >>> transform(Point(0, 0, 0), lambda x: x + 1, include_z=None)
    <POINT Z (1 1 1)>

    With interleaved=False, the call signature of the transformation is different:

    >>> transform(LineString([(1, 2), (3, 4)]), lambda x, y: (x + 1, y), interleaved=False)
    <LINESTRING (2 2, 4 4)>

    Or with a z coordinate:

    >>> transform(Point(0, 0, 0), lambda x, y, z: (x + 1, y, z + 2), interleaved=False, include_z=True)
    <POINT Z (1 0 2)>

    Using pyproj >= 2.1, the following example will reproject Shapely geometries
    from EPSG 4326 to EPSG 32618:

    >>> from pyproj import Transformer
    >>> transformer = Transformer.from_crs(4326, 32618, always_xy=True)
    >>> transform(Point(-75, 50), transformer.transform, interleaved=False)
    <POINT (500000 5538630.703)>
    """
    geometry_arr = np.array(geometry, dtype=np.object_)  # makes a copy
    if include_z is None:
        has_z = shapely.has_z(geometry_arr)
        result = np.empty_like(geometry_arr)
        result[has_z] = transform(
            geometry_arr[has_z], transformation, True, interleaved
        )
        result[~has_z] = transform(
            geometry_arr[~has_z], transformation, False, interleaved
        )
    else:
        coordinates = lib.get_coordinates(geometry_arr, include_z, False)
        if interleaved:
            new_coordinates = transformation(coordinates)
        else:
            new_coordinates = np.asarray(
                transformation(*coordinates.T), dtype=np.float64
            ).T
        # check the array to yield understandable error messages
        if not isinstance(new_coordinates, np.ndarray) or new_coordinates.ndim != 2:
            raise ValueError(
                "The provided transformation did not return a two-dimensional numpy array"
            )
        if new_coordinates.dtype != np.float64:
            raise ValueError(
                "The provided transformation returned an array with an unexpected "
                f"dtype ({new_coordinates.dtype})"
            )
        if new_coordinates.shape != coordinates.shape:
            # if the shape is too small we will get a segfault
            raise ValueError(
                "The provided transformation returned an array with an unexpected "
                f"shape ({new_coordinates.shape})"
            )
        result = lib.set_coordinates(geometry_arr, new_coordinates)
    if result.ndim == 0 and not isinstance(geometry, np.ndarray):
        return result.item()
    return result


def transform_resize(
    geom: shapely.Geometry,
    transformation,
    include_z: Optional[bool] = False,
    interleaved: bool = True,
):
    """Apply a function to the coordinates of a Geometry object, optionally changing the number
    of coordinate pairs.

    This function differs with `transform` in the following ways:

    - The number of coordinates per Geometry is allowed to change.
    - The input parameters to the transformation function are (tuple of) float, not array.
    - It only accepts scalar Geometry objects, not arrays.

    We recommend only using this function only if 1) the number of coordinates in your Geometry
    should change or 2) the transformation can't be made to accept numpy arrays.

    Parameters
    ----------
    geom : Geometry
    transformation : function
        A function that maps x, y, and optionally z to output xp, yp, zp.
        The input parameters are preferably tuples of float, however if that raises
        a TypeError the input parameters are floats. The output shall be of the same
        type as the input. When doing tuples, the output tuples may be shorter or longer
        than the input tuples.
    include_z : bool, optional, default False
        If False, always return 2D geometries.
        If True, the data being passed to the
        transformation function will include the third dimension
        (if a geometry has no third dimension, the z-coordinates
        will be NaN). If None, will infer the dimensionality using
        ``has_z``. Note that this inference
        can be unreliable with empty geometries or NaN coordinates: for a
        guaranteed result, it is recommended to specify ``include_z`` explicitly.

    See Also
    --------
    has_z : Returns a copy of a geometry array with a function applied to its
        coordinates.
    transform : Transform arrays of Geometry objects.

    Examples
    --------
    >>> from shapely import LineString, Point

    Reduce a linestring to only its first 2 points:

    >>> transform_resize(LineString([(2, 2), (4, 4), (6, 6)]), lambda x, y: x[:2], y:2))
    <LINESTRING (2 2, 4 4)>

    Transform a point using a lambda function that accepts only scalars:

    >>> transform_resize(Point(0, 0), lambda x, y: (x + 1, y + 2))
    <POINT (1 2)>
    """
    assert include_z is None
    assert interleaved is False
    if geom.is_empty:
        return geom
    geom_type = shapely.get_type_id(geom)
    if geom_type in (
        GeometryType.POINT,
        GeometryType.LINESTRING,
        GeometryType.LINEARRING,
    ):
        # First we try to apply func to x, y, z sequences. When func is
        # optimized for sequences, this is the fastest, though zipping
        # the results up to go back into the geometry constructors adds
        # extra cost.
        try:
            return type(geom)(zip(*transformation(*zip(*geom.coords))))
        # A func that assumes x, y, z are single values will likely raise a
        # TypeError, in which case we'll try again.
        except TypeError:
            return type(geom)([transformation(*c) for c in geom.coords])
    elif geom_type == GeometryType.POLYGON:
        # First we try to apply func to x, y, z sequences. When func is
        # optimized for sequences, this is the fastest, though zipping
        # the results up to go back into the geometry constructors adds
        # extra cost.
        try:
            shell = type(geom.exterior)(
                zip(*transformation(*zip(*geom.exterior.coords)))
            )
            holes = list(
                type(ring)(zip(*transformation(*zip(*ring.coords))))
                for ring in geom.interiors
            )
            return type(geom)(shell, holes)

        # A func that assumes x, y, z are single values will likely raise a
        # TypeError, in which case we'll try again.
        except TypeError:
            shell = type(geom.exterior)(
                [transformation(*c) for c in geom.exterior.coords]
            )
            holes = list(
                type(ring)([transformation(*c) for c in ring.coords])
                for ring in geom.interiors
            )
            return type(geom)(shell, holes)
    elif geom_type in (
        GeometryType.MULTIPOINT,
        GeometryType.MULTILINESTRING,
        GeometryType.MULTIPOLYGON,
        GeometryType.GEOMETRYCOLLECTION,
    ):
        return type(geom)(
            [
                transform_resize(
                    part, transformation, include_z=include_z, interleaved=interleaved
                )
                for part in geom.geoms
            ]
        )
    else:
        raise TypeError(f"Type {geom_type} not recognized")


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
