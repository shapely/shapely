Changelog
=========

Version 0.10 (unreleased)
------------------------

**Major enhancements**

* Addition of ``nearest`` and ``nearest_all`` functions to ``STRtree`` for
  GEOS >= 3.6 to find the nearest neighbors (#272).

**API Changes**

* STRtree default leaf size is now 10 instead of 5, for somewhat better performance
  under normal conditions (#286)
* Removes ``VALID_PREDICATES`` set from ``pygeos.strtree`` package; these can be constructed
  in downstream libraries using the ``pygeos.strtree.BinaryPredicate`` enum.

**Added GEOS functions**

* Addition of a ``contains_properly`` function (#267)
* Addition of a ``polygonize`` function (#275)
* Addition of a ``segmentize`` function for GEOS >= 3.10 (#299)

**Bug fixes**

* Fixed portability issue for ARM architecture (#293)

**Acknowledgments**

Thanks to everyone who contributed to this release!
People with a "+" by their names contributed a patch for the first time.

* Brendan Ward
* Casper van der Wel
* Joris Van den Bossche


Version 0.9 (2021-01-23)
------------------------

**Major enhancements**

* Addition of ``prepare`` function that generates a GEOS prepared geometry which is stored on
  the Geometry object itself. All binary predicates (except ``equals``) make use of this.
  Helper functions ``destroy_prepared`` and ``is_prepared`` are also available. (#92, #252)
* Use previously prepared geometries within ``STRtree`` ``query`` and ``query_bulk``
  functions if available (#246)
* Official support for Python 3.9 and numpy 1.20 (#278, #279)
* Drop support for Python 3.5 (#211)
* Added support for pickling to ``Geometry`` objects (#190)
* The ``apply`` function for coordinate transformations and the ``set_coordinates``
  function now support geometries with z-coordinates (#131)
* Addition of Cython and internal PyGEOS C API to enable easier development of internal
  functions (previously all significant internal functions were developed in C).
  Added a Cython-implemented ``get_parts`` function (#51)

**API Changes**

* Geometry and counting functions (``get_num_coordinates``,
  ``get_num_geometries``, ``get_num_interior_rings``, ``get_num_points``) now return 0
  for ``None`` input values instead of -1 (#218)
* ``intersection_all`` and ``symmetric_difference_all`` now ignore None values
  instead of returning None if any value is None (#249)
* ``union_all`` now returns None (instead of ``GEOMETRYCOLLECTION EMPTY``) if
  all input values are None (#249)
* The default axis of ``union_all``, ``intersection_all``, ``symmetric_difference_all``,
  and ``coverage_union_all`` can now reduce over multiple axes. The default changed from the first
  axis (``0``) to all axes (``None``) (#266)
* Argument in ``line_interpolate_point`` and ``line_locate_point``
  was renamed from ``normalize`` to ``normalized`` (#209)
* Addition of ``grid_size`` parameter to specify fixed-precision grid for ``difference``,
  ``intersection``, ``symmetric_difference``, ``union``, and ``union_all`` operations for
  GEOS >= 3.9 (#276)

**Added GEOS functions**

* Release the GIL for ``is_geometry()``, ``is_missing()``, and
  ``is_valid_input()`` (#207)
* Addition of a ``is_ccw()`` function for GEOS >= 3.7 (#201)
* Addition of a ``minimum_clearance`` function for GEOS >= 3.6.0 (#223)
* Addition of a ``offset_curve`` function (#229)
* Addition of a ``relate_pattern`` function (#245)
* Addition of a ``clip_by_rect`` function (#273)
* Addition of a ``reverse`` function for GEOS >= 3.7 (#254)
* Addition of ``get_precision`` to get precision of a geometry and ``set_precision``
  to set the precision of a geometry (may round and reduce coordinates) (#257)

**Bug fixes**

* Fixed internal GEOS error code detection for ``get_dimensions`` and ``get_srid`` (#218)
* Limited the length of geometry repr to 80 characters (#189)
* Fixed error handling in ``line_locate_point`` for incorrect geometry
  types, now actually requiring line and point geometries (#216)
* Addition of ``get_parts`` function to get individual parts of an array of multipart
  geometries (#197)
* Ensure that ``python setup.py clean`` removes all previously Cythonized and compiled
  files (#239)
* Handle GEOS beta versions  (#262)

**Acknowledgments**

Thanks to everyone who contributed to this release!
People with a "+" by their names contributed a patch for the first time.

* Brendan Ward
* Casper van der Wel
* Joris Van den Bossche
* Mike Taves


Version 0.8 (2020-09-06)
------------------------

**Highlights of this release**

* Handle multi geometries in ``boundary`` (#188)
* Handle empty points in to_wkb by conversion to POINT (nan, nan) (#179)
* Prevent segfault in to_wkt (and repr) with empty points in multipoints (#171)
* Fixed bug in ``multilinestrings()``, it now accepts linearrings again (#168)
* Release the GIL to allow for multithreading in functions that do not
  create geometries (#144) and in the STRtree ``query_bulk()`` method (#174)
* Addition of a ``frechet_distance()`` function for GEOS >= 3.7 (#144)
* Addition of ``coverage_union()`` and ``coverage_union_all()`` functions
  for GEOS >= 3.8 (#142)
* Fixed segfaults when adding empty geometries to the STRtree (#147)
* Addition of ``include_z=True`` keyword in the ``get_coordinates()`` function
  to get 3D coordinates (#178)
* Addition of a ``build_area()`` function for GEOS >= 3.8 (#141)
* Addition of a ``normalize()`` function (#136)
* Addition of a ``make_valid()`` function for GEOS >= 3.8 (#107)
* Addition of a ``get_z()`` function for GEOS >= 3.7 (#175)
* Addition of a ``relate()`` function (#186)
* The ``get_coordinate_dimensions()`` function was renamed to
  ``get_coordinate_dimension()`` for consistency with GEOS (#176)
* Addition of ``covers``, ``covered_by``, ``contains_properly`` predicates
  to STRtree ``query`` and ``query_bulk`` (#157)

**Acknowledgments**

Thanks to everyone who contributed to this release!
People with a "+" by their names contributed a patch for the first time.

* Brendan Ward
* Casper van der Wel
* Joris Van den Bossche
* Krishna Chaitanya +
* Martin Fleischmann +
* Tom Clancy +


Version 0.7 (2020-03-18)
------------------------

**Highlights of this release**

* STRtree improvements for spatial indexing:
  * Directly include predicate evaluation in ``STRtree.query()`` (#87)
  * Query multiple input geometries (spatial join style) with ``STRtree.query_bulk`` (#108)
* Addition of a ``total_bounds()`` function (#107)
* Geometries are now hashable, and can be compared with ``==`` or ``!=`` (#102)
* Fixed bug in ``create_collections()`` with wrong types (#86)
* Fixed a reference counting bug in STRtree (#97, #100)
* Start of a benchmarking suite using ASV (#96)
* This is the first release that will provide wheels!

**Acknowledgments**

Thanks to everyone who contributed to this release!
People with a "+" by their names contributed a patch for the first time.

* Brendan Ward +
* Casper van der Wel
* Joris Van den Bossche
* Mike Taves +


Version 0.6 (2020-01-31)
------------------------

Highlights of this release:

* Addition of the STRtree class for spatial indexing (#58)
* Addition of a ``bounds`` function (#69)
* A new ``from_shapely`` function to convert Shapely geometries to pygeos.Geometry (#61)
* Reintroduction of the ``shared_paths`` function (#77)

Contributors:

* Casper van der Wel
* Joris Van den Bossche
* mattijn +


Version 0.5 (2019-10-25)
------------------------

Highlights of this release:

* Moved to the pygeos GitHub organization.
* Addition of functionality to get and transform all coordinates (eg for reprojections or affine transformations) [#44]
* Ufuncs for converting to and from the WKT and WKB formats [#45]
* ``equals_exact`` has been added [PR #57]


Version 0.4 (2019-09-16)
------------------------

This is a major release of PyGEOS and the first one with actual release notes. Most important features of this release are:

* ``buffer`` and ``haussdorff_distance`` were completed  [#15]
* ``voronoi_polygons`` and ``delaunay_triangles`` have been added [#17]
* The PyGEOS documentation is now mostly complete and available on http://pygeos.readthedocs.io .
* The concepts of "empty" and "missing" geometries have been separated. The ``pygeos.Empty`` and ``pygeos.NaG`` objects has been removed. Empty geometries are handled the same as normal geometries. Missing geometries are denoted by ``None`` and are handled by every pygeos function. ``NaN`` values cannot be used anymore to denote missing geometries. [PR #36]
* Added ``pygeos.__version__`` and ``pygeos.geos_version``. [PR #43]

