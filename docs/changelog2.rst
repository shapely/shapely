


Changelog
=========


Version 2.0.0 (2022-??-??)
--------------------------

Refactor of the internals
^^^^^^^^^^^^^^^^^^^^^^^^^

No longer using ctypes, but C extension

Top-level element-wise functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Not only all methods on the Geometry classes, but also some methods from submodules (eg ..)

* Vectorized constructor functions
* Optionally output to a user-specified array (``out`` keyword argument) when constructing
  geometries from ``indices`` (#380).
* Enable bulk construction of geometries with different number of coordinates
  by optionally taking index arrays in all creation functions (#230, #322, #326, #346).


API changes (deprecated in 1.8)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

See :ref:`migration` for more details on how to update your code.

Recommended to first upgrade to Shapely 1.8 and resolve all deprecation
warnings before upgrading to Shapely 2.0.

Summary of changes:

* Geometries are now immutable and hashable.
* Multi-part geometries no longer behave as "sequences". This means that they
  no longer have a length, are not iterable and indexable anymore. Use ``.geoms``
  attribute instead to access the parts.
* Geometries no longer directly implement the numpy array interface. To
  convert to an array of coordinates, use `.coords` instead
  (`np.asarray(geom.coords)`).
* Consistent creation of empty geometries (for exampe ``Polygon()`` now
  actually creates an empty Polygon instead of an empty geometry collection).
* The following attributes and methods on the Geometry classes were
  deprecated and are now removed:
  * ``array_interface()`` and ``ctypes``
  * ``asShape()``, and the adapters classes to create geometry-like proxy
    objects (use ``shape()`` instead).
  * ``empty()`` method

Breaking API changes
^^^^^^^^^^^^^^^^^^^^

Some additional backwards incompatible API changes were included in Shapely
2.0 that were not yet deprecated in Shapely 1.8:

* The default of the ``preserve_topology`` keyword of ``simplify()`` changed
  to True (#1392).
* A ``GeometryCollection`` that consists of all empty sub-geometries now
  returns those empty geometries in ``.geoms`` (instead of returning an empty
  list) (#1420).
* The unused ``shape_factory()`` method and ``HeterogeneousGeometrySequence``
  class are removed (#1421).
* The undocumted ``__geom__`` attribute is removed. To access the raw GEOS pointer,
  the ``_geom`` attribute is still present (1417).

New features
^^^^^^^^^^^^

More informative repr with (truncated) WKT
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Support for fixed precision model for geometries and in overlay functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* `set_precision` to conform a geometry to a certain grid size, and this will then also be used by subsequent overlay methods
* `grid_size` keyword in overlay methods (`intersection`, `union`, `difference`, etc) 


* Addition of ``get_precision`` to get precision of a geometry and ``set_precision``
  to set the precision of a geometry (may round and reduce coordinates) (#257)
* Addition of ``grid_size`` parameter to specify fixed-precision grid for ``difference``,
  ``intersection``, ``symmetric_difference``, ``union``, and ``union_all`` operations for
  GEOS >= 3.9 (#276)

Releasing the GIL
~~~~~~~~~~~~~~~~~

* Release the GIL to allow for multithreading in functions that do not
  create geometries (#144) and in the STRtree ``query_bulk()`` method (#174)
* Released the GIL in all geometry creation functions (#310, #326).

STRtree improvements
~~~~~~~~~~~~~~~~~~~~

* Specifying a `predicate` in the STRtree queries 
* Bulk queries


* STRtree improvements for spatial indexing:
  * Directly include predicate evaluation in ``STRtree.query()`` (#87)
  * Query multiple input geometries (spatial join style) with ``STRtree.query_bulk`` (#108)
* Fixed ``STRtree`` creation to allow querying the tree in a multi-threaded
  context (#361).

Bindings for new GEOS functionalities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Several (new) functions from GEOS are now exposed in Shapely:

* ``haussdorff_distance`` and ``frechet_distance()``
* ``contains_properly``
* ``extract_unique_points``
* ``reverse``
* ``build_area()`` (GEOS >= 3.8)
* ``minimum_bounding_circle`` and ``minimum_bounding_radius`` (GEOS >= 3.8)
* ``coverage_union()`` and ``coverage_union_all()`` (GEOS >= 3.8)
* ``segmentize`` (GEOS >= 3.10)
* ``dwithin`` (GEOS >= 3.10)

In addition some aliases for existing methods have been added to provide a
method name consistent with GEOS or PostGIS:

- ``line_interpolate_point`` (``interpolate``)
- ``line_locate_point`` (``project``)
- ``offset_curve`` (``parallel_offset``)
- ``point_on_surface`` (``representative_point``)
- ``oriented_envelope`` (``minimum_rotated_rectangle``)
- ``delauney_triangles`` (``ops.triangulate``)
- ``voronoi_polygons`` (``ops.voronoi_diagram``)
- ``shortest_line`` (``ops.nearest_points``)
- ``is_valid_reason`` (``validation.explain_validity``)


Getting information / parts / coordinates from geometries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A set of GEOS getter functions are now also exposed to inspect geometries:
`get_dimensions`, `get_coordinate_dimension`, `get_srid`, `get_num_points`,
`get_num_interior_rings`, `get_num_geometries`, `get_num_coordinates`,
`get_precision`.


To extract parts: `get_geometry`, `get_exterior_ring`, `get_interior_ring`,
`get_parts`, `get_rings`, `get_point`, `get_coordinates`, `get_x`, `get_y`,
`get_z`



* Addition of ``get_parts`` function to get individual parts of an array of multipart
  geometries (#197)
* Added the option to return the geometry index in ``get_coordinates`` (#318).
* Added the ``get_rings`` function, similar as ``get_parts`` but specifically
  to extract the rings of Polygon geometries (#342).


Prepared geometries
~~~~~~~~~~~~~~~~~~~

Prepared geometries are now no longer separate objects, but geometry objects itself
can be prepared (this makes the `shapely.prepared` module superfluous).

* Addition of ``prepare`` function that generates a GEOS prepared geometry which is stored on
  the Geometry object itself. All binary predicates (except ``equals``) make use of this.
  Helper functions ``destroy_prepared`` and ``is_prepared`` are also available. (#92, #252)


GeoJSON IO
~~~~~~~~~~

* Added GeoJSON input/output capabilities (``pygeos.from_geojson``,
  ``pygeos.to_geojson``) for GEOS >= 3.10 (#413).

Other improvements
~~~~~~~~~~~~~~~~~~

* Added ``pygeos.force_2d`` and ``pygeos.force_3d`` to change the dimensionality of
  the coordinates in a geometry (#396).

* Addition of a ``total_bounds()`` function (#107)

* Performance improvement in constructing LineStrings or LinearRings from
  numpy arrays for GEOS >= 3.10 (#436)

* Updated ``box`` ufunc to use internal C function for creating polygon
  (about 2x faster) and added ``ccw`` parameter to create polygon in
  counterclockwise (default) or clockwise direction (#308).

* Start of a benchmarking suite using ASV (#96)

Utilities

* Added ``pygeos.testing.assert_geometries_equal`` (#401).


* Added ``pygeos.empty`` to create a geometry array pre-filled with None or
  with empty geometries (#381).

Bug fixes
~~~~~~~~~

TODO: check if this are also bug fixes in Shapely 2.0 compared to 1.8

* Return True instead of False for LINEARRING geometries in ``is_closed`` (#379).
* Fixed the WKB serialization of 3D empty points for GEOS >= 3.9.0 (#392).
* Fixed the WKT serialization of single part 3D empty geometries for GEOS >= 3.9.0 (#402).
* Fixed the WKT serialization of multipoints with empty points for GEOS >= 3.9.0 (#392).
* Fixed a segfault when getting coordinates from empty points in GEOS 3.8.0 (#415).

* Fixed portability issue for ARM architecture (#293)
* Fixed segfault in ``linearrings`` and ``box`` when constructing a geometry with nan
  coordinates (#310).
* Fixed segfault in ``polygons`` (with holes) when None was provided.
* Fixed memory leak in ``polygons`` when non-linearring input was provided.

* Handle empty points in to_wkb by conversion to POINT (nan, nan) (#179)
* Prevent segfault in to_wkt (and repr) with empty points in multipoints (#171)
* Fixed segfaults when adding empty geometries to the STRtree (#147)


**Acknowledgments**

Thanks to everyone who contributed to this release!
People with a "+" by their names contributed a patch for the first time.

(TODO update from actual git log, this is only from the PyGEOS changelog notes)

* Brendan Ward +
* Casper van der Wel +
* Joris Van den Bossche
* Mike Taves
* Tanguy Ophoff +
* James Myatt +
* Krishna Chaitanya +
* Martin Fleischmann +
* Tom Clancy +
* mattijn +
