Geometry
========

The ``pygeos.Geometry`` class is the central datatype in PyGEOS. 
An instance of ``Geometry`` is a container of the actual GEOSGeometry object.
The Geometry object keeps track of the underlying GEOSGeometry and
lets the python garbage collector free its memory when it is not
used anymore.

Geometry objects are immutable. This means that after constructed, they cannot
be changed in place. Every PyGEOS operation will result in a new object being returned.

Construction
~~~~~~~~~~~~

For convenience, the ``Geometry`` class can be constructed with a WKT (Well-Known Text)
or WKB (Well-Known Binary) representation of a geometry:

.. code:: python

  >>> from pygeos import Geometry
  >>> point_1 = Geometry("POINT (5.2 52.1)")
  >>> point_2 = Geometry(b"\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\xf0?")

A more efficient way of constructing geometries is by making use of the (vectorized)
functions described in :mod:`pygeos.creation`.

Pickling
~~~~~~~~

Geometries can be serialized using pickle:

  >>> import pickle
  >>> pickled = pickle.dumps(point_1)
  >>> pickle.loads(point_1)
  <pygeos.Geometry POINT (5.2 52.1)>

.. warning:: Pickling will convert linearrings to linestrings and it may convert empty points.
             See :func:`pygeos.io.to_wkb` for a complete list of limitations.

Hashing
~~~~~~~

Geometries can be used as elements in sets or as keys in dictionaries.
Python uses a technique called *hashing* for lookups in these datastructures.
PyGEOS generates this hash from the WKB representation.
Therefore, geometries are equal if and only if their WKB representations are equal.

.. code:: python

  >>> point_3 = Geometry("POINT (5.2 52.1)")
  >>> {point_1, point_2, point_3}
  {<pygeos.Geometry POINT (5.2 52.1)>, <pygeos.Geometry POINT (1 1)>}

.. warning:: Due to limitations of WKB, linearrings will equal linestrings if they contain the exact same points.
             Also, different kind of empty points will be considered equal. See :func:`pygeos.io.to_wkb`.

Comparing two geometries directly is also supported.
This is the same as using :func:`pygeos.predicates.equals_exact` with a ``tolerance`` value of zero.

  >>> point_1 == point_2
  False
  >>> point_1 != point_2
  True


Properties
~~~~~~~~~~

Geometry objects have neither properties nor methods.
Instead, use the functions listed below to obtain information about geometry objects.

.. automodule:: pygeos.geometry
   :members:
   :exclude-members: GeometryType
   :special-members:
   :inherited-members:
   :show-inheritance:
