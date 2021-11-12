Changelog
=========


Version 0.12 (unreleased)
-------------------------

**Distribution**

* Distribute binary wheels for Apple Silicon architecture
  (arm64 and universal2) (#427).
* Removed 32-bit architecture wheels for
  Python 3.10 (#427).
* All binary wheels now have GEOS 3.10.1. See https://github.com/libgeos/geos/blob/main/NEWS
  for the changes (#422).


**Major enhancements**

* Added ``pygeos.dwithin`` for GEOS >= 3.10 (#417).
* Added GeoJSON input/output capabilities (``pygeos.from_geojson``, 
  ``pygeos.to_geojson``) for GEOS >= 3.10 (#413).

**API Changes**

* ...

**Bug fixes**

* ...


**Acknowledgments**

Thanks to everyone who contributed to this release!
People with a "+" by their names contributed a patch for the first time.

* Brendan Ward
* Casper van der Wel
* Joris Van den Bossche


Version 0.11.1 (2021-10-30)
---------------------------

**Distribution**

* Distribute binary wheels for Python 3.10 (#416).
* All binary wheels now have GEOS 3.10.0. See https://github.com/libgeos/geos/blob/main/NEWS
  for the changes (#416).


**Major enhancements**

* Optionally output to a user-specified array (``out`` keyword argument) when constructing
  geometries from ``indices`` (#380).
* Added ``pygeos.empty`` to create a geometry array pre-filled with None or
  with empty geometries (#381).
* Added ``pygeos.force_2d`` and ``pygeos.force_3d`` to change the dimensionality of
  the coordinates in a geometry (#396).
* Added ``pygeos.testing.assert_geometries_equal`` (#401).

**API Changes**

* The default behaviour of ``pygeos.set_precision`` is now to always return valid geometries.
  Before, the default was ``preserve_topology=False`` which caused confusion because
  it mapped to GEOS_PREC_NO_TOPO (the new 'pointwise').
  At the same time, GEOS < 3.10 implementation was not entirely correct so that some geometries
  did and some did not preserve topology with this mode. Now, the new ``mode`` argument controls
  the behaviour and the ``preserve_topology`` argument is deprecated (#410).
* When constructing a linearring through ``pygeos.linearrings`` or a polygon through 
  ``pygeos.polygons`` now a ``ValueError`` is raised (instead of a ``GEOSException``)
  if the ring contains less than 4 coordinates including ring closure (#378).
* Removed deprecated ``normalize`` keyword argument in ``pygeos.line_locate_point`` and
  ``pygeos.line_interpolate_point`` (#410).

**Bug fixes**

* Return True instead of False for LINEARRING geometries in ``is_closed`` (#379).
* Fixed the WKB serialization of 3D empty points for GEOS >= 3.9.0 (#392).
* Fixed the WKT serialization of single part 3D empty geometries for GEOS >= 3.9.0 (#402).
* Fixed the WKT serialization of multipoints with empty points for GEOS >= 3.9.0 (#392).
* Fixed a segfault when getting coordinates from empty points in GEOS 3.8.0 (#415).

**Acknowledgments**

Thanks to everyone who contributed to this release!
People with a "+" by their names contributed a patch for the first time.

* Brendan Ward
* Casper van der Wel
* Joris Van den Bossche


Version 0.10.2 (2021-08-23)
---------------------------

**Distribution**

Unittests are now included in the pygeos distribution. Run them by 1) installing
``pytest`` (or ``pygeos[test]``) and 2) invoking ``pytest --pyargs pygeos.tests``.

We started using a new tool for building binary wheels: ``cibuildwheel``. This
resulted into the following improvements in the distributed binary wheels:

* Windows binary wheels now contain mangled DLLs, which avoids conflicts
  with other GEOS versions present on the system (a.k.a. 'DLL hell') (#365).
* Windows binary wheels now contain the Microsoft Visual C++ Runtime Files
  (msvcp140.dll) for usage on systems without the C++ redistributable (#365).
* Linux x86_64 and i686 wheels are now built using the manylinux2010 image
  instead of manylinux1 (#365).
* Linux aarch64 wheels are now available for Python 3.9 (manylinux2014, #365).

**Bug fixes**

* Fixed operations on geometry arrays containing NULL instead of None.
  These occur for instance by using ``numpy.empty_like`` (#371)

**Acknowledgements**

Thanks to everyone who contributed to this release!
People with a "+" by their names contributed a patch for the first time.

* Brendan Ward
* Casper van der Wel
* Joris Van den Bossche


Version 0.10.1 (2021-07-06)
---------------------------

**Bug fixes**

* Fixed the ``box`` and ``set_precision`` functions with numpy 1.21 (#367).
* Fixed ``STRtree`` creation to allow querying the tree in a multi-threaded
  context (#361).

**Acknowledgements**

Thanks to everyone who contributed to this release!
People with a "+" by their names contributed a patch for the first time.

* Brendan Ward
* Casper van der Wel
* Joris Van den Bossche


Version 0.10 (2021-05-18)
-------------------------

**Major enhancements**

* Addition of ``nearest`` and ``nearest_all`` functions to ``STRtree`` for
  GEOS >= 3.6 to find the nearest neighbors (#272).
* Enable bulk construction of geometries with different number of coordinates
  by optionally taking index arrays in all creation functions (#230, #322, #326, #346).
* Released the GIL in all geometry creation functions (#310, #326).
* Added the option to return the geometry index in ``get_coordinates`` (#318).
* Added the ``get_rings`` function, similar as ``get_parts`` but specifically
  to extract the rings of Polygon geometries (#342).
* Updated ``box`` ufunc to use internal C function for creating polygon
  (about 2x faster) and added ``ccw`` parameter to create polygon in
  counterclockwise (default) or clockwise direction (#308).
* Added ``to_shapely`` and improved performance of ``from_shapely`` in the case
  GEOS versions are different (#312).

**API Changes**

* STRtree default leaf size is now 10 instead of 5, for somewhat better performance
  under normal conditions (#286)
* Deprecated ``VALID_PREDICATES`` set from ``pygeos.strtree`` package; these can be constructed
  in downstream libraries using the ``pygeos.strtree.BinaryPredicate`` enum.
  This will be removed in a future release.
* ``points``, ``linestrings``, ``linearrings``, and ``polygons`` now return a ``GEOSException``
  instead of a ``ValueError`` or ``TypeError`` for invalid input (#310, #326).
* Addition of ``on_invalid`` parameter to ``from_wkb`` and ``from_wkt`` to
  optionally return invalid WKB geometries as ``None``.
* Removed the (internal) function ``lib.polygons_without_holes`` and renamed
  ``lib.polygons_with_holes`` to ``lib.polygons`` (#326).
* ``polygons`` will now return an empty polygon for `None` inputs (#346).
* Removed compatibility with Python 3.5 (#341).


**Added GEOS functions**

* Addition of a ``contains_properly`` function (#267)
* Addition of a ``polygonize`` function (#275)
* Addition of a ``polygonize_full`` function (#298)
* Addition of a ``segmentize`` function for GEOS >= 3.10 (#299)
* Addition of ``oriented_envelope`` and ``minimum_rotated_rectangle`` functions (#314)
* Addition of ``minimum_bounding_circle`` and ``minimum_bounding_radius`` functions for GEOS >= 3.8 (#315)
* Addition of a ``shortest_line`` ("nearest points") function (#334)

**Bug fixes**

* Fixed portability issue for ARM architecture (#293)
* Fixed segfault in ``linearrings`` and ``box`` when constructing a geometry with nan
  coordinates (#310).
* Fixed segfault in ``polygons`` (with holes) when None was provided.
* Fixed memory leak in ``polygons`` when non-linearring input was provided.

**Acknowledgments**

Thanks to everyone who contributed to this release!
People with a "+" by their names contributed a patch for the first time.

* Brendan Ward
* Casper van der Wel
* Joris Van den Bossche
* Martin Fleischmann
* Mike Taves
* Tanguy Ophoff +
* James Myatt +


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

