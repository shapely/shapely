.. _migration-pygeos:

=====================
Migrating from PyGEOS
=====================

The PyGEOS package was merged with Shapely in December 2021 and will be
released as part of Shapely 2.0. No further development will take place for
the PyGEOS package (except for providing up to date packages until Shapely
2.0 is released).

Therefore, everybody using PyGEOS is highly recommended to migrate to Shapely
2.0.

Generally speaking, this should be a smooth experience because all
functionality of PyGEOS was added to Shapely. All vectorized functions
available in ``pygeos`` have been added to the top-level ``shapely`` module,
with only minor differences (see below). Migrating from PyGEOS to Shapely 2.0
can thus be done by replacing the ``pygeos`` import and module calls::

    import pygeos
    polygon = pygeos.box(0, 0, 2, 2)
    points = pygeos.points(...)
    pygeos.contains(polygon, points)

Using Shapely 2.0, this can now be written as::

    import shapely
    polygon = shapely.box(0, 0, 2, 2)
    points = shapely.points(...)
    shapely.contains(polygon, points)

In addition, you now also have the scalar interface of Shapely which wasn't
implemented in PyGEOS.

Differences between PyGEOS and Shapely 2.0
==========================================

STRtree API changes
-------------------

Functionality-wise, everything from ``pygeos.STRtree`` is available in
Shapely 2.0. But while merging into Shapely, some methods have been changed
or merged:

- The ``query()`` and ``query_bulk()`` methods have been merged into a single
  ``query()`` method. The ``query()`` method now accepts an array of
  geometries as well in addition to a single geometry, and in that case it
  will return 2D array of indices.

  It should thus be a matter of replacing ``query_bulk`` with ``query`` in
  your code.

  See :meth:`.STRtree.query` for more details.

- The ``nearest()`` method was changed to return an array of the same shape
  as the input geometries. Thus, for a scalar geometry it now returns a
  single integer index (instead of a (2, 1) array), and for an array of
  geometries it now returns a 1D array of indices ((n,) array instead of a
  (2, n) array).

  See :meth:`.STRtree.nearest` for more details.

- The ``nearest_all()`` method has been replaced with ``query_nearest()``.
  For an array of geometries, the output is the same, but when passing a
  scalar geometry as input, the method now returns a 1D array instead of a 2D
  array (consistent with ``query()``).

  In addition, this method gained the new ``exclusive`` and ``all_matches``
  keywords (with defaults preserving existing behaviour from PyGEOS). See
  :meth:`.STRtree.query_nearest` for more details.


Other differences
-----------------

- The ``pygeos.Geometry(..)`` constructor has not been retained in Shapely
  (the class exists as base class, but the constructor is not callable). Use
  one of the subclasses, or ``shapely.from_wkt(..)``, instead.
- The ``apply()`` function was renamed to ``transform()``.
- The ``tolerance`` keyword of the ``segmentize()`` function was renamed to
  ``max_segment_length``.
- The ``quadsegs`` keyword of the ``buffer()`` and ``offset_curve()``
  functions was renamed to ``quad_segs``.
- The ``preserve_topology`` keyword of ``simplify()`` now defaults to
  ``True`` instead of ``False``.
- The behaviour of ``union_all()`` / ``intersection_all()`` / ``symmetric_difference_all``
  was changed to return an empty GeometryCollection for an empty or all-None
  sequence as input (instead of returning None).
- The ``radius`` keyword of the ``buffer()`` function was renamed to ``distance``.
