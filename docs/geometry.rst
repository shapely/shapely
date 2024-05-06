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
functions described in :ref:`ref-creation`.

Pickling
~~~~~~~~

Geometries can be serialized using pickle:

  >>> import pickle
  >>> from shapely import Point
  >>> pickled = pickle.dumps(Point(1, 1))
  >>> pickle.loads(pickled)
  <POINT (1 1)>

.. warning:: Pickling will convert linearrings to linestrings.
             See :func:`shapely.to_wkb` for a complete list of limitations.

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
             See :func:`shapely.to_wkb`.

Comparing two geometries directly is also supported.
This is the same as using :func:`shapely.equals_exact` with a ``tolerance`` value of zero.

  >>> point_1 == point_2
  False
  >>> point_1 == point_3
  True
  >>> point_1 != point_2
  True

Formatting
~~~~~~~~~~

Geometries can be formatted to strings using properties, functions,
or a Python format specification.

The most convenient is to use ``.wkb_hex`` and ``.wkt`` properties.

.. code:: python

  >>> from shapely import Point, to_wkb, to_wkt, to_geojson
  >>> pt = Point(-169.910918, -18.997564)
  >>> pt.wkb_hex
  0101000000CF6A813D263D65C0BDAAB35A60FF32C0
  >>> pt.wkt
  POINT (-169.910918 -18.997564)

More output options can be found using using :func:`~shapely.to_wkb`,
:func:`~shapely.to_wkt`, and :func:`~shapely.to_geojson` functions.

.. code:: python

  >>> to_wkb(pt, hex=True, byte_order=0)
  0000000001C0653D263D816ACFC032FF605AB3AABD
  >>> to_wkt(pt, rounding_precision=3)
  POINT (-169.911 -18.998)
  >>> print(to_geojson(pt, indent=2))
  {
    "type": "Point",
    "coordinates": [
      -169.910918,
      -18.997564
    ]
  }

A format specification may also be used to control the format and precision.

.. code:: python

    >>> print(f"Cave near {pt:.3f}")
    Cave near POINT (-169.911 -18.998)
    >>> print(f"or hex-encoded as {pt:x}")
    or hex-encoded as 0101000000cf6a813d263d65c0bdaab35a60ff32c0

Shapely has a format specification inspired from Python's
:ref:`python:formatspec`, described next.

Semantic for format specification
---------------------------------

.. productionlist:: format-spec
  format_spec: [0][.`precision`][`type`]
  precision: `digit`+
  digit: "0"..."9"
  type: "f" | "F" | "g" | "G" | "x" | "X"

Format types ``'f'`` and ``'F'`` are to use a fixed-point notation, which is
activated by setting GEOS' trim option off.
The upper case variant converts ``nan`` to ``NAN`` and ``inf`` to ``INF``.

Format types ``'g'`` and ``'G'`` are to use a "general format",
where unnecessary digits are trimmed. This notation is activated by setting
GEOS' trim option on. The upper case variant is similar to
``'F'``, and may also display an upper-case ``"E"`` if scientific notation
is required. Note that this representation may be different for GEOS 3.10.0
and later, which does not use scientific notation.

For numeric outputs ``'f'`` and ``'g'``, the precision is optional, and if not
specified, rounding precision will be disabled showing full precision.

Format types ``'x'`` and ``'X'`` show a hex-encoded string representation of
WKB or Well-Known Binary, with the case of the output matched the
case of the format type character.

.. _canonical-form:

Canonical form
--------------
When operations are applied on geometries the result is returned according to
some conventions.

In most cases, geometries will be returned in "mild" canonical form. There is no
goal to keep this form stable, so it is expected to change in future versions of
GEOS:

- the coordinates of exterior rings follow a clockwise orientation and interior
  rings have a counter-clockwise orientation. This is the opposite of the OGC
  specifications because the choice was made before this was included in the
  standard.
- the starting point of rings can be changed in the output, but the exact order
  is undefined and should not be relied upon
- the order of geometry types in a collection can be changed, but the order is
  undefined

When :func:`~shapely.normalize` is used, the "strict" canonical form is applied.
This type of normalization is meant to be stable, so changes to it will be
avoided if possible:

- the coordinates of exterior rings follow a clockwise orientation and interior
  rings have a counter-clockwise orientation
- the starting point of rings is lower left
- elements in collections are ordered by geometry type: by descending dimension
  and multi-types first (MultiPolygon, Polygon, MultiLineString, LineString,
  MultiPoint, Point). Multiple elements from the same type are ordered from
  right to left and from top to bottom.

It is important to note that input geometries do not have to follow these
conventions.
