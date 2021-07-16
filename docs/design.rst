============
Design Notes
============

Shapely provides classes that implement, more or less, the interfaces in the
OGC's simple feature access specification [1]_. The classes are defined in
similarly named modules under ``shapely.geometry``: ``Point`` is in
``shapely.geometry.point``, ``MultiPolygon`` is in
``shapely.geometry.multipolygon``. These classes derive from
``shapely.geometry.base.BaseGeometry``. The simple features methods of
``BaseGeometry`` call functions registered in a class variable ``impl``. For
example, ``BaseGeometry.area`` calls ``BaseGeometry.impl['area']``.

The default registry is in the ``shapely.impl`` module. Its items are classes
that operate on single geometric objects or pairs of geometric objects.
Pluggability is a goal of this design, but we're not there yet. Some work needs
to be done before anybody can use CGAL as a Shapely backend.

In sum, Shapely's stack is 4 layers:

* Python geometry classes in ``shapely.geometry``
* An implementation registry: an abstraction that permits alternate geometry
  engines, even a mix of geometry engines. The default is in ``shapely.impl``.
* The GEOS implementations of methods for the registry in ``shapely.geos``.
* libgeos: algorithms written in C++.

.. [1] John R. Herring, Ed.,
   “OpenGIS Implementation Specification for Geographic information - Simple
   feature access - Part 1: Common architecture,” Oct. 2006.

