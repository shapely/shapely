from . import lib, Geometry
import numpy as np

__all__ = ["apply", "count_coordinates", "get_coordinates", "set_coordinates"]


def apply(geometry, transformation, include_z=False):
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
        Whether to include the third dimension in the coordinates array
        that is passed to the `transformation` function. If True, and a
        geometry has no third dimension, the z-coordinates passed to the
        function will be NaN.

    Examples
    --------
    >>> apply(Geometry("POINT (0 0)"), lambda x: x + 1)
    <pygeos.Geometry POINT (1 1)>
    >>> apply(Geometry("LINESTRING (2 2, 4 4)"), lambda x: x * [2, 3])
    <pygeos.Geometry LINESTRING (4 6, 8 12)>
    >>> apply(None, lambda x: x) is None
    True
    >>> apply([Geometry("POINT (0 0)"), None], lambda x: x).tolist()
    [<pygeos.Geometry POINT (0 0)>, None]

    By default, the third dimension is ignored:

    >>> apply(Geometry("POINT Z (0 0 0)"), lambda x: x + 1)
    <pygeos.Geometry POINT (1 1)>
    >>> apply(Geometry("POINT Z (0 0 0)"), lambda x: x + 1, include_z=True)
    <pygeos.Geometry POINT Z (1 1 1)>
    """
    geometry_arr = np.array(geometry, dtype=np.object_)  # makes a copy
    coordinates = lib.get_coordinates(geometry_arr, include_z)
    new_coordinates = transformation(coordinates)
    # check the array to yield understandable error messages
    if not isinstance(new_coordinates, np.ndarray):
        raise ValueError("The provided transformation did not return a numpy array")
    if new_coordinates.dtype != np.float64:
        raise ValueError(
            "The provided transformation returned an array with an unexpected "
            "dtype ({})".format(new_coordinates.dtype)
        )
    if new_coordinates.shape != coordinates.shape:
        # if the shape is too small we will get a segfault
        raise ValueError(
            "The provided transformation returned an array with an unexpected "
            "shape ({})".format(new_coordinates.shape)
        )
    geometry_arr = lib.set_coordinates(geometry_arr, new_coordinates)
    if geometry_arr.ndim == 0 and not isinstance(geometry, np.ndarray):
        return geometry_arr.item()
    return geometry_arr


def count_coordinates(geometry):
    """Counts the number of coordinate pairs in a geometry array.

    Parameters
    ----------
    geometry : Geometry or array_like

    Examples
    --------
    >>> count_coordinates(Geometry("POINT (0 0)"))
    1
    >>> count_coordinates(Geometry("LINESTRING (2 2, 4 4)"))
    2
    >>> count_coordinates(None)
    0
    >>> count_coordinates([Geometry("POINT (0 0)"), None])
    1
    """
    return lib.count_coordinates(np.asarray(geometry, dtype=np.object_))


def get_coordinates(geometry, include_z=False):
    """Gets coordinates from a geometry array as an array of floats.

    The shape of the returned array is (N, 2), with N being the number of
    coordinate pairs. With the default of `include_z=False`, three-dimensional
    data is ignored. When specifying `include_z=True`, the shape of the
    returned array is (N, 3).

    Parameters
    ----------
    geometry : Geometry or array_like
    include_z : bool, default False
        Whether to include the third dimension in the output. If True, and a
        geometry has no third dimension, the z-coordinates will be NaN.

    Examples
    --------
    >>> get_coordinates(Geometry("POINT (0 0)")).tolist()
    [[0.0, 0.0]]
    >>> get_coordinates(Geometry("LINESTRING (2 2, 4 4)")).tolist()
    [[2.0, 2.0], [4.0, 4.0]]
    >>> get_coordinates(None)
    array([], shape=(0, 2), dtype=float64)

    By default the third dimension is ignored:

    >>> get_coordinates(Geometry("POINT Z (0 0 0)")).tolist()
    [[0.0, 0.0]]
    >>> get_coordinates(Geometry("POINT Z (0 0 0)"), include_z=True).tolist()
    [[0.0, 0.0, 0.0]]
    """
    return lib.get_coordinates(np.asarray(geometry, dtype=np.object_), include_z)


def set_coordinates(geometry, coordinates):
    """Returns a copy of a geometry array with different coordinates.

    If the coordinates array has shape (N, 2), all returned geometries
    will be two-dimensional, and the third dimension will be discarded,
    if present. If the coordinates array has shape (N, 3), the returned
    geometries preserve the dimensionality of the input geometries.

    Parameters
    ----------
    geometry : Geometry or array_like
    coordinates: array_like

    Examples
    --------
    >>> set_coordinates(Geometry("POINT (0 0)"), [[1, 1]])
    <pygeos.Geometry POINT (1 1)>
    >>> set_coordinates([Geometry("POINT (0 0)"), Geometry("LINESTRING (0 0, 0 0)")], [[1, 2], [3, 4], [5, 6]]).tolist()
    [<pygeos.Geometry POINT (1 2)>, <pygeos.Geometry LINESTRING (3 4, 5 6)>]
    >>> set_coordinates([None, Geometry("POINT (0 0)")], [[1, 2]]).tolist()
    [None, <pygeos.Geometry POINT (1 2)>]

    Third dimension of input geometry is discarded if coordinates array does
    not include one:

    >>> set_coordinates(Geometry("POINT Z (0 0 0)"), [[1, 1]])
    <pygeos.Geometry POINT (1 1)>
    >>> set_coordinates(Geometry("POINT Z (0 0 0)"), [[1, 1, 1]])
    <pygeos.Geometry POINT Z (1 1 1)>
    """
    geometry_arr = np.asarray(geometry, dtype=np.object_)
    coordinates = np.atleast_2d(np.asarray(coordinates)).astype(np.float64)
    n_coords = lib.count_coordinates(geometry_arr)
    if coordinates.ndim != 2:
        raise ValueError(
            "The coordinate array should have dimension of 2 "
            "(has {})".format(coordinates.ndim)
        )
    if (
        (coordinates.shape[0] != lib.count_coordinates(geometry_arr))
        or (coordinates.shape[1] not in {2, 3})
    ):
        raise ValueError(
            "The coordinate array has an invalid shape {}".format(coordinates.shape)
        )
    lib.set_coordinates(geometry_arr, coordinates)
    if geometry_arr.ndim == 0 and not isinstance(geometry, np.ndarray):
        return geometry_arr.item()
    return geometry_arr
