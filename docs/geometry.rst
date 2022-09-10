Geometry
========

Shapely geometry classes, such as ``shapely.Point``, are the central data types
in Shapely.  Each geometry class extends the ``shapely.Geometry`` base class,
which is a container of the underlying GEOS geometry object, to provide geometry
type-specific attributes and behavior.  The ``Geometry`` object keeps track of
the underlying GEOS geometry and lets the python garbage collector free its
memory when it is not used anymore.

Geometry objects are immutable. This means that after constructed, they cannot
be changed in place. Every Shapely operation will result in a new object being
returned.

Geometry types
~~~~~~~~~~~~~~

.. currentmodule:: shapely

.. autosummary::
    :toctree: reference/

    Point
    LineString
    LinearRing
    Polygon
    MultiPoint
    MultiLineString
    MultiPolygon
    GeometryCollection

Construction
~~~~~~~~~~~~

Geometries can be constructed directly using Shapely geometry classes:

.. code:: python

  >>> from shapely import Point, LineString
  >>> Point(5.2, 52.1)
  <POINT (5.2 52.1)>
  >>> LineString([(0, 0), (1, 2)])
  <LINESTRING (0 0, 1 2)>

Geometries can also be constructed from a WKT (Well-Known Text)
or WKB (Well-Known Binary) representation:

.. code:: python

  >>> from shapely import from_wkb, from_wkt
  >>> from_wkt("POINT (5.2 52.1)")
  <POINT (5.2 52.1)>
  >>> from_wkb(b"\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\xf0?")
  <POINT (1 1)>

A more efficient way of constructing geometries is by making use of the (vectorized)
functions described in :mod:`shapely.creation`.

Pickling
~~~~~~~~

Geometries can be serialized using pickle:

  >>> import pickle
  >>> from shapely import Point
  >>> pickled = pickle.dumps(Point(1, 1))
  >>> pickle.loads(pickled)
  <POINT (1 1)>

.. warning:: Pickling will convert linearrings to linestrings.
             See :func:`shapely.io.to_wkb` for a complete list of limitations.

Hashing
~~~~~~~

Geometries can be used as elements in sets or as keys in dictionaries.
Python uses a technique called *hashing* for lookups in these datastructures.
Shapely generates this hash from the WKB representation.
Therefore, geometries are equal if and only if their WKB representations are equal.

.. code:: python

  >>> from shapely import Point
  >>> point_1 = Point(5.2, 52.1)
  >>> point_2 = Point(1, 1)
  >>> point_3 = Point(5.2, 52.1)
  >>> {point_1, point_2, point_3}
  {<POINT (1 1)>, <POINT (5.2 52.1)>}

.. warning:: Due to limitations of WKB, linearrings will equal linestrings if they contain the exact same points.
             See :func:`shapely.io.to_wkb`.

Comparing two geometries directly is also supported.
This is the same as using :func:`shapely.predicates.equals_exact` with a ``tolerance`` value of zero.

  >>> point_1 == point_2
  False
  >>> point_1 == point_3
  True
  >>> point_1 != point_2
  True
