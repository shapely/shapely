.. _migration:

==============================
Migrating to Shapely 1.8 / 2.0
==============================

Shapely 1.8.0 is a transitional version introducing several warnings in
preparation of the upcoming changes in 2.0.0.

Shapely 2.0.0 will be a major release with a refactor of the internals with
considerable performance improvements (based on the developments in the
`PyGEOS <https://github.com/pygeos/pygeos>`__ package), along with several
breaking changes.

This guide gives an overview of the most important changes with details
on what will change in 2.0.0, how we warn for this in 1.8.0, and how
you can update your code to be future-proof.

For more background, see
`RFC 1: Roadmap for Shapely 2.0 <https://github.com/shapely/shapely-rfc/pull/1>`__.

.. contents:: Table of Contents
  :backlinks: none
  :local:


Geometry objects will become immutable
======================================

Geometry objects will become immutable in version 2.0.0.

In Shapely 1.x, some of the geometry classes are mutable, meaning that you
can change their coordinates in-place. Illustrative code::

    >>> from shapely.geometry import LineString
    >>> line = LineString([(0,0), (2, 2)])
    >>> print(line)
    LINESTRING (0 0, 2 2)

    >>> line.coords = [(0, 0), (10, 0), (10, 10)]
    >>> print(line)
    LINESTRING (0 0, 10 0, 10 10)

In Shapely 1.8, this will start raising a warning::

    >>> line.coords = [(0, 0), (10, 0), (10, 10)]
    ShapelyDeprecationWarning: Setting the 'coords' to mutate a Geometry
    in place is deprecated, and will not be possible any more in Shapely 2.0

and starting with version 2.0.0, all geometry objects will become immutable.
As a consequence, they will also become hashable and therefore usable as, for
example, dictionary keys.

**How do I update my code?** There is no direct alternative for mutating the
coordinates of an existing geometry, except for creating a new geometry
object with the new coordinates.


Setting custom attributes
-------------------------

Another consequence of the geometry objects becoming immutable is that
assigning custom attributes, which currently works, will no longer be possible.

Currently you can do::

    >>> line.name = "my_geometry"
    >>> line.name
    'my_geometry'

In Shapely 1.8, this will start raising a warning, and will raise an
AttributeError in Shapely 2.0.

**How do I update my code?** There is no direct alternative for adding custom
attributes to geometry objects. You can use other Python data structures such as
(GeoJSON-like) dictionaries or GeoPandas' GeoDataFrames to store attributes
alongside geometry features. 

Multi-part geometries will no longer be "sequences" (length, iterable, indexable)
=================================================================================

In Shapely 1.x, multi-part geometries (MultiPoint, MultiLineString,
MultiPolygon and GeometryCollection) implement a part of the "sequence"
python interface (making them list-like). This means you can iterate through
the object to get the parts, index into the object to get a specific part,
and ask for the number of parts with the ``len()`` method.

Some examples of this with Shapely 1.x:

    >>> from shapely.geometry import Point, MultiPoint
    >>> mp = MultiPoint([(1, 1), (2, 2), (3, 3)])
    >>> print(mp)
    MULTIPOINT (1 1, 2 2, 3 3)
    >>> for part in mp:
    ...     print(part)
    POINT (1 1)
    POINT (2 2)
    POINT (3 3)
    >>> print(mp[1])
    POINT (2 2)
    >>> len(mp)
    3
    >>> list(mp)
    [<shapely.geometry.point.Point at 0x7f2e0912bf10>,
     <shapely.geometry.point.Point at 0x7f2e09fed820>,
     <shapely.geometry.point.Point at 0x7f2e09fed4c0>]

Starting with Shapely 1.8, all the examples above will start raising a
deprecation warning. For example:

    >>> for part in mp:
    ...     print(part)
    ShapelyDeprecationWarning: Iteration over multi-part geometries is deprecated
    and will be removed in Shapely 2.0. Use the `geoms` property to access the
    constituent parts of a multi-part geometry.
    POINT (1 1)
    POINT (2 2)
    POINT (3 3)

In Shapely 2.0, all those examples will raise an error.

**How do I update my code?** To access the geometry parts of a multi-part
geometry, you can use the ``.geoms`` attribute, as the warning indicates.

The examples above can be updated to::

    >>> for part in mp.geoms:
    ...     print(part)
    POINT (1 1)
    POINT (2 2)
    POINT (3 3)
    >>> print(mp.geoms[1])
    POINT (2 2)
    >>> len(mp.geoms)
    3
    >>> list(mp.geoms)
    [<shapely.geometry.point.Point at 0x7f2e0912bf10>,
     <shapely.geometry.point.Point at 0x7f2e09fed820>,
     <shapely.geometry.point.Point at 0x7f2e09fed4c0>]

The single-part geometries (Point, LineString, Polygon) already didn't
support those features, and for those classes there is no change in behaviour
for this aspect.


Interopability with NumPy and the array interface
=================================================

Conversion of the coordinates to (NumPy) arrays
-----------------------------------------------

Shapely provides an array interface to have easy access to the coordinates as,
for example, NumPy arrays (:ref:`manual section <array-interface>`).

A small example::

    >>> line = LineString([(0, 0), (1, 1), (2, 2)])
    >>> import numpy as np
    >>> np.asarray(line)
    array([[0., 0.],
           [1., 1.],
           [2., 2.]])

In addition, there are also the explicit ``array_interface()`` method and
``ctypes`` attribute to get access to the coordinates as array data:

    >>> line.ctypes
    <shapely.geometry.linestring.c_double_Array_6 at 0x7f75261eb740>
    >>> line.array_interface()
    {'version': 3,
     'typestr': '<f8',
     'data': <shapely.geometry.linestring.c_double_Array_6 at 0x7f752664ae40>,
     'shape': (3, 2)}

This functionality is available for Point, LineString, LinearRing and MultiPoint.

For more robust interoperability with NumPy, this array interface will be removed
from those geometry classes, and limited to the ``coords``. 

Starting with Shapely 1.8, converting a geometry object to a NumPy array
directly will start raising a warning::

    >>> np.asarray(line)
    ShapelyDeprecationWarning: The array interface is deprecated and will no longer
    work in Shapely 2.0. Convert the '.coords' to a NumPy array instead.
    array([[0., 0.],
           [1., 1.],
           [2., 2.]])

**How do I update my code?** To convert a geometry to a NumPy array, you can
convert the ``.coords`` attribute instead::

    >>> line.coords
    <shapely.coords.CoordinateSequence at 0x7f2e09e88d60>
    >>> np.array(line.coords)
    array([[0., 0.],
           [1., 1.],
           [2., 2.]])

The ``array_interface()`` method and ``ctypes`` attribute will be removed in
Shapely 2.0, but since Shapely will start requiring NumPy as a dependency,
you can use NumPy or its array interface directly. Check the NumPy docs on
the :py:attr:`ctypes <numpy:numpy.ndarray.ctypes>` attribute
or the :ref:`array interface <numpy:arrays.interface>` for more details.

Creating NumPy arrays of geometry objects
-----------------------------------------

Shapely geometry objects can be stored in NumPy arrays using the ``object``
dtype. In general, one could create such an array from a list of geometries
as follows::

    >>> from shapely.geometry import Point
    >>> arr = np.array([Point(0, 0), Point(1, 1), Point(2, 2)])
    >>> arr
    array([<shapely.geometry.point.Point object at 0x7fb798407cd0>,
           <shapely.geometry.point.Point object at 0x7fb7982831c0>,
           <shapely.geometry.point.Point object at 0x7fb798283b80>],
          dtype=object)

The above works for point geometries, but because in Shapely 1.x, some
geometry types are sequence-like (see above), NumPy can try to "unpack" them
when creating an array. Therefore, for more robust creation of a NumPy array
from a list of geometries, it's generally recommended to this in a two-step
way (first creating an empty array and then filling it)::

    geoms = [Point(0, 0), Point(1, 1), Point(2, 2)]
    arr = np.empty(len(geoms), dtype="object")
    arr[:] = geoms

This code snippet results in the same array as the example above, and works
for all geometry types and Shapely/NumPy versions. 

However, starting with Shapely 1.8, the above code will show deprecation
warnings that cannot be avoided (depending on the geometry type, NumPy tries
to access the array interface of the objects or check if an object is
iterable or has a length, and those operations are all deprecated now. The
end result is still correct, but the warnings appear nonetheless).
Specifically in this case, it is fine to ignore those warnings (and the only
way to make them go away)::

    import warnings
    from shapely.errors import ShapelyDeprecationWarning

    geoms = [Point(0, 0), Point(1, 1), Point(2, 2)]
    arr = np.empty(len(geoms), dtype="object")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
        arr[:] = geoms

In Shapely 2.0, the geometry objects will no longer be sequence like and
those deprecation warnings will be removed (and thus the ``filterwarnings``
will no longer be necessary), and creation of NumPy arrays will generally be
more robust.

If you maintain code that depends on Shapely, and you want to have it work
with multiple versions of Shapely, the above code snippet provides a context
manager that can be copied into your project::

    import contextlib
    import shapely
    import warnings
    from packaging import version  # https://packaging.pypa.io/

    SHAPELY_GE_20 = version.parse(shapely.__version__) >= version.parse("2.0a1")

    try:
        from shapely.errors import ShapelyDeprecationWarning as shapely_warning
    except ImportError:
        shapely_warning = None

    if shapely_warning is not None and not SHAPELY_GE_20:
        @contextlib.contextmanager
        def ignore_shapely2_warnings():
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=shapely_warning)
                yield
    else:
        @contextlib.contextmanager
        def ignore_shapely2_warnings():
            yield

This can then be used when creating NumPy arrays (be careful to *only* use it
for this specific purpose, and not generally suppress those warnings)::

    geoms = [...]
    arr = np.empty(len(geoms), dtype="object")
    with ignore_shapely2_warnings():
        arr[:] = geoms


Consistent creation of empty geometries
=======================================

Shapely 1.x is inconsistent in creating empty geometries between various
creation methods. A small example for an empty Polygon geometry:

.. code-block:: python

    # Using an empty constructor results in a GeometryCollection
    >>> from shapely.geometry import Polygon
    >>> g1 = Polygon()
    >>> type(g1)
    <class 'shapely.geometry.polygon.Polygon'>
    >>> g1.wkt
    GEOMETRYCOLLECTION EMPTY

    # Converting from WKT gives a correct empty polygon
    >>> from shapely import wkt
    >>> g2 = wkt.loads("POLYGON EMPTY")
    >>> type(g2)
    <class 'shapely.geometry.polygon.Polygon'>
    >>> g2.wkt
    POLYGON EMPTY

Shapely 1.8 does not yet change this inconsistent behaviour, but starting
with Shapely 2.0, the different methods will always consistently give an
empty geometry object of the correct type, instead of using an empty
GeometryCollection as "generic" empty geometry object.

**How do I update my code?** Those cases that will change don't raise a
warning, but you will need to update your code if you rely on the fact that
empty geometry objects are of the GeometryCollection type. Use the
``.is_empty`` attribute for robustly checking if a geometry object is an
empty geometry.

In addition, the WKB serialization methods will start supporting empty
Points (using ``"POINT (NaN NaN)"`` to represent an empty point).


Other deprecated functionality
==============================

There are some other various functions and methods deprecated in Shapely 1.8
as well:

- The adapters to create geometry-like proxy objects with coordinates stored
  outside Shapely geometries are deprecated and will be removed in Shapely
  2.0 (e.g. created using ``asShape()``). They have little to no benefit
  compared to the normal geometry classes, as thus you can convert to your
  data to a normal geometry object instead. Use the ``shape()`` function
  instead to convert a GeoJSON-like dict to a Shapely geometry.

- The ``empty()`` method on a geometry object is deprecated.

- The ``shapely.ops.cascaded_union`` function is deprecated. Use
  ``shapely.ops.unary_union`` instead, which internally already uses a cascaded union operation for better performance.
