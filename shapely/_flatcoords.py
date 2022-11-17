"""
This modules provides a conversion to / from a ragged (or "jagged") array
representation of the geometries.

A ragged array is an irregular array of arrays of which each element can have
a different length. As a result, such an array cannot be represented as a
standard, rectangular nD array.
The coordinates of geometries can be represented as arrays of arrays of
coordinate pairs (possibly multiple levels of nesting, depending on the
geometry type).


Geometries, as a ragged array of coordinates, can be efficiently represented
as contiguous arrays of coordinates provided that there is another data
structure that keeps track of which range of coordinate values corresponds
to a given geometry. This can be done using offsets, counts, or indices.

This module currently implements offsets into the coordinates array. This
is the ragged array representation defined by the the Apache Arrow project
as "variable size list array" (https://arrow.apache.org/docs/format/Columnar.html#variable-size-list-layout).
See for example https://cfconventions.org/Data/cf-conventions/cf-conventions-1.9/cf-conventions.html#representations-features
for different options.

The exact usage of the Arrow list array with varying degrees of nesting for the
different geometry types is defined by the GeoArrow project:
https://github.com/geoarrow/geoarrow

"""
import numpy as np

from . import creation
from ._geometry import GeometryType, get_parts, get_rings, get_type_id
from .coordinates import get_coordinates
from .predicates import is_empty

__all__ = ["to_ragged_array", "from_ragged_array"]


# # GEOS -> coords/offset arrays (to_ragged_array)


def _get_arrays_point(arr, include_z):
    # only one array of coordinates
    coords = get_coordinates(arr, include_z=include_z)

    # empty points are represented by NaNs
    empties = is_empty(arr)
    if empties.any():
        indices = np.nonzero(empties)[0]
        indices = indices - np.arange(len(indices))
        coords = np.insert(coords, indices, np.nan, axis=0)

    return coords, ()


def _indices_to_offsets(indices, n):
    offsets = np.insert(np.bincount(indices).cumsum(), 0, 0)
    if len(offsets) != n + 1:
        # last geometries might be empty or missing
        offsets = np.pad(
            offsets,
            (0, n + 1 - len(offsets)),
            "constant",
            constant_values=offsets[-1],
        )
    return offsets


def _get_arrays_multipoint(arr, include_z):
    # explode/flatten the MultiPoints
    _, part_indices = get_parts(arr, return_index=True)
    # the offsets into the multipoint parts
    offsets = _indices_to_offsets(part_indices, len(arr))

    # only one array of coordinates
    coords = get_coordinates(arr, include_z=include_z)

    return coords, (offsets,)


def _get_arrays_linestring(arr, include_z):
    # the coords and offsets into the coordinates of the linestrings
    coords, indices = get_coordinates(arr, return_index=True, include_z=include_z)
    offsets = _indices_to_offsets(indices, len(arr))

    return coords, (offsets,)


def _get_arrays_multilinestring(arr, include_z):
    # explode/flatten the MultiLineStrings
    arr_flat, part_indices = get_parts(arr, return_index=True)
    # the offsets into the multilinestring parts
    offsets2 = _indices_to_offsets(part_indices, len(arr))

    # the coords and offsets into the coordinates of the linestrings
    coords, indices = get_coordinates(arr_flat, return_index=True, include_z=include_z)
    offsets1 = np.insert(np.bincount(indices).cumsum(), 0, 0)

    return coords, (offsets1, offsets2)


def _get_arrays_polygon(arr, include_z):
    # explode/flatten the Polygons into Rings
    arr_flat, ring_indices = get_rings(arr, return_index=True)
    # the offsets into the exterior/interior rings of the multipolygon parts
    offsets2 = _indices_to_offsets(ring_indices, len(arr))

    # the coords and offsets into the coordinates of the rings
    coords, indices = get_coordinates(arr_flat, return_index=True, include_z=include_z)
    offsets1 = np.insert(np.bincount(indices).cumsum(), 0, 0)

    return coords, (offsets1, offsets2)


def _get_arrays_multipolygon(arr, include_z):
    # explode/flatten the MultiPolygons
    arr_flat, part_indices = get_parts(arr, return_index=True)
    # the offsets into the multipolygon parts
    offsets3 = _indices_to_offsets(part_indices, len(arr))

    # explode/flatten the Polygons into Rings
    arr_flat2, ring_indices = get_rings(arr_flat, return_index=True)
    # the offsets into the exterior/interior rings of the multipolygon parts
    offsets2 = np.insert(np.bincount(ring_indices).cumsum(), 0, 0)

    # the coords and offsets into the coordinates of the rings
    coords, indices = get_coordinates(arr_flat2, return_index=True, include_z=include_z)
    offsets1 = np.insert(np.bincount(indices).cumsum(), 0, 0)

    return coords, (offsets1, offsets2, offsets3)


def to_ragged_array(geometries, include_z=False):
    """
    Converts geometries to a ragged array representation using a contiguous
    array of coordinates and offset arrays.

    This function converts an array of geometries to a ragged array
    (i.e. irregular array of arrays) of coordinates, represented in memory
    using a single contiguous array of the coordinates, and
    up to 3 offset arrays that keep track where each sub-array
    starts and ends.

    This follows the in-memory layout of the variable size list arrays defined
    by Apache Arrow, as specified for geometries by the GeoArrow project:
    https://github.com/geoarrow/geoarrow.

    Parameters
    ----------
    geometries : array_like
        Array of geometries (1-dimensional).
    include_z : bool, default False
        If True, include the third dimension in the output. If a geometry
        has no third dimension, the z-coordinates will be NaN.

    Returns
    -------
    tuple of (geometry_type, coords, offsets)
        geometry_type : GeometryType
            The type of the input geometries (required information for
            roundtrip).
        coords : np.ndarray
            Contiguous array of shape (n, 2) or (n, 3) of all coordinates
            of all input geometries.
        offsets: tuple of np.ndarray
            Offset arrays that make it possible to reconstruct the
            geometries from the flat coordinates array. The number of
            offset arrays depends on the geometry type. See
            https://github.com/geoarrow/geoarrow/blob/main/format.md
            for details.

    Notes
    -----
    Mixed singular and multi geometry types of the same basic type are
    allowed (e.g., Point and MultiPoint) and all singular types will be
    treated as multi types.
    GeometryCollections and other mixed geometry types are not supported.

    See also
    --------
    from_ragged_array

    Examples
    --------
    Consider a Polygon with one hole (interior ring):

    >>> import shapely
    >>> polygon = shapely.Polygon(
    ...     [(0, 0), (10, 0), (10, 10), (0, 10)],
    ...     holes=[[(2, 2), (3, 2), (2, 3)]]
    ... )
    >>> polygon
    <POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (2 2, 3 2, 2 3, 2 2))>

    This polygon can be thought of as a list of rings (first ring is the
    exterior ring, subsequent rings are the interior rings), and each ring
    as a list of coordinate pairs. This is very similar to how GeoJSON
    represents the coordinates:

    >>> import json
    >>> json.loads(shapely.to_geojson(polygon))["coordinates"]
    [[[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [0.0, 0.0]],
     [[2.0, 2.0], [3.0, 2.0], [2.0, 3.0], [2.0, 2.0]]]

    This function will return a similar list of lists of lists, but
    using a single contiguous array of coordinates, and multiple arrays of
    offsets:

    >>> geometry_type, coords, offsets = shapely.to_ragged_array([polygon])
    >>> geometry_type
    <GeometryType.POLYGON: 3>
    >>> coords
    array([[ 0.,  0.],
           [10.,  0.],
           [10., 10.],
           [ 0., 10.],
           [ 0.,  0.],
           [ 2.,  2.],
           [ 3.,  2.],
           [ 2.,  3.],
           [ 2.,  2.]])

    >>> offsets
    (array([0, 5, 9]), array([0, 2]))

    As an example how to interpret the offsets: the i-th ring in the
    coordinates is represented by ``offsets[0][i]`` to ``offsets[0][i+1]``:

    >>> exterior_ring_start, exterior_ring_end = offsets[0][0], offsets[0][1]
    >>> coords[exterior_ring_start:exterior_ring_end]
    array([[ 0.,  0.],
           [10.,  0.],
           [10., 10.],
           [ 0., 10.],
           [ 0.,  0.]])

    """
    geom_types = np.unique(get_type_id(geometries))
    # ignore missing values (type of -1)
    geom_types = geom_types[geom_types >= 0]

    if len(geom_types) == 1:
        typ = GeometryType(geom_types[0])
        if typ == GeometryType.POINT:
            coords, offsets = _get_arrays_point(geometries, include_z)
        elif typ == GeometryType.LINESTRING:
            coords, offsets = _get_arrays_linestring(geometries, include_z)
        elif typ == GeometryType.POLYGON:
            coords, offsets = _get_arrays_polygon(geometries, include_z)
        elif typ == GeometryType.MULTIPOINT:
            coords, offsets = _get_arrays_multipoint(geometries, include_z)
        elif typ == GeometryType.MULTILINESTRING:
            coords, offsets = _get_arrays_multilinestring(geometries, include_z)
        elif typ == GeometryType.MULTIPOLYGON:
            coords, offsets = _get_arrays_multipolygon(geometries, include_z)
        else:
            raise ValueError(f"Geometry type {typ.name} is not supported")

    elif len(geom_types) == 2:
        if set(geom_types) == {GeometryType.POINT, GeometryType.MULTIPOINT}:
            typ = GeometryType.MULTIPOINT
            coords, offsets = _get_arrays_multipoint(geometries, include_z)
        elif set(geom_types) == {GeometryType.LINESTRING, GeometryType.MULTILINESTRING}:
            typ = GeometryType.MULTILINESTRING
            coords, offsets = _get_arrays_multilinestring(geometries, include_z)
        elif set(geom_types) == {GeometryType.POLYGON, GeometryType.MULTIPOLYGON}:
            typ = GeometryType.MULTIPOLYGON
            coords, offsets = _get_arrays_multipolygon(geometries, include_z)
        else:
            raise ValueError(
                "Geometry type combination is not supported "
                f"({[GeometryType(t).name for t in geom_types]})"
            )
    else:
        raise ValueError(
            "Geometry type combination is not supported "
            f"({[GeometryType(t).name for t in geom_types]})"
        )

    return typ, coords, offsets


# # coords/offset arrays -> GEOS (from_ragged_array)


def _point_from_flatcoords(coords):
    result = creation.points(coords)

    # Older versions of GEOS (<= 3.9) don't automatically convert NaNs
    # to empty points -> do manually
    empties = np.isnan(coords).all(axis=1)
    if empties.any():
        result[empties] = creation.empty(1, geom_type=GeometryType.POINT).item()

    return result


def _multipoint_from_flatcoords(coords, offsets):
    # recreate points
    points = creation.points(coords)

    # recreate multipoints
    multipoint_parts = np.diff(offsets)
    multipoint_indices = np.repeat(np.arange(len(multipoint_parts)), multipoint_parts)

    result = np.empty(len(offsets) - 1, dtype=object)
    result = creation.multipoints(points, indices=multipoint_indices, out=result)
    result[multipoint_parts == 0] = creation.empty(
        1, geom_type=GeometryType.MULTIPOINT
    ).item()

    return result


def _linestring_from_flatcoords(coords, offsets):
    # recreate linestrings
    linestring_n = np.diff(offsets)
    linestring_indices = np.repeat(np.arange(len(linestring_n)), linestring_n)

    result = np.empty(len(offsets) - 1, dtype=object)
    result = creation.linestrings(coords, indices=linestring_indices, out=result)
    result[linestring_n == 0] = creation.empty(
        1, geom_type=GeometryType.LINESTRING
    ).item()
    return result


def _multilinestrings_from_flatcoords(coords, offsets1, offsets2):
    # recreate linestrings
    linestrings = _linestring_from_flatcoords(coords, offsets1)

    # recreate multilinestrings
    multilinestring_parts = np.diff(offsets2)
    multilinestring_indices = np.repeat(
        np.arange(len(multilinestring_parts)), multilinestring_parts
    )

    result = np.empty(len(offsets2) - 1, dtype=object)
    result = creation.multilinestrings(
        linestrings, indices=multilinestring_indices, out=result
    )
    result[multilinestring_parts == 0] = creation.empty(
        1, geom_type=GeometryType.MULTILINESTRING
    ).item()

    return result


def _polygon_from_flatcoords(coords, offsets1, offsets2):
    # recreate rings
    ring_lengths = np.diff(offsets1)
    ring_indices = np.repeat(np.arange(len(ring_lengths)), ring_lengths)
    rings = creation.linearrings(coords, indices=ring_indices)

    # recreate polygons
    polygon_rings_n = np.diff(offsets2)
    polygon_indices = np.repeat(np.arange(len(polygon_rings_n)), polygon_rings_n)
    result = np.empty(len(offsets2) - 1, dtype=object)
    result = creation.polygons(rings, indices=polygon_indices, out=result)
    result[polygon_rings_n == 0] = creation.empty(
        1, geom_type=GeometryType.POLYGON
    ).item()

    return result


def _multipolygons_from_flatcoords(coords, offsets1, offsets2, offsets3):
    # recreate polygons
    polygons = _polygon_from_flatcoords(coords, offsets1, offsets2)

    # recreate multipolygons
    multipolygon_parts = np.diff(offsets3)
    multipolygon_indices = np.repeat(
        np.arange(len(multipolygon_parts)), multipolygon_parts
    )
    result = np.empty(len(offsets3) - 1, dtype=object)
    result = creation.multipolygons(polygons, indices=multipolygon_indices, out=result)
    result[multipolygon_parts == 0] = creation.empty(
        1, geom_type=GeometryType.MULTIPOLYGON
    ).item()

    return result


def from_ragged_array(geometry_type, coords, offsets=None):
    """
    Creates geometries from a contiguous array of coordinates
    and offset arrays.

    This function creates geometries from the ragged array representation
    as returned by ``to_ragged_array``.

    This follows the in-memory layout of the variable size list arrays defined
    by Apache Arrow, as specified for geometries by the GeoArrow project:
    https://github.com/geoarrow/geoarrow.

    See :func:`to_ragged_array` for more details.

    Parameters
    ----------
    geometry_type : GeometryType
        The type of geometry to create.
    coords : np.ndarray
        Contiguous array of shape (n, 2) ro (n, 3) of all coordinates
        for the geometries.
    offsets: tuple of np.ndarray
        Offset arrays that allow to reconstruct the geometries based on the
        flat coordinates array. The number of offset arrays depends on the
        geometry type. See
        https://github.com/geoarrow/geoarrow/blob/main/format.md for details.

    Returns
    -------
    np.ndarray
        Array of geometries (1-dimensional).

    See Also
    --------
    to_ragged_array

    """
    if geometry_type == GeometryType.POINT:
        assert offsets is None or len(offsets) == 0
        return _point_from_flatcoords(coords)
    if geometry_type == GeometryType.LINESTRING:
        return _linestring_from_flatcoords(coords, *offsets)
    if geometry_type == GeometryType.POLYGON:
        return _polygon_from_flatcoords(coords, *offsets)
    elif geometry_type == GeometryType.MULTIPOINT:
        return _multipoint_from_flatcoords(coords, *offsets)
    elif geometry_type == GeometryType.MULTILINESTRING:
        return _multilinestrings_from_flatcoords(coords, *offsets)
    elif geometry_type == GeometryType.MULTIPOLYGON:
        return _multipolygons_from_flatcoords(coords, *offsets)
    else:
        raise ValueError(f"Geometry type {geometry_type.name} is not supported")
