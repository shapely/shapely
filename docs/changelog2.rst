


Changelog
=========


Version 2.0.0 (2022-??-??)
--------------------------

The Shapely 2.0.0 version is a major release featuring a complete refactor of
the internals and new vectorized (element-wise) array operations providing
considerable performance improvements (based on the developments in the
`PyGEOS <https://github.com/pygeos/pygeos>`__ package), along with several
breaking API changes and many feature improvements.

For more background, see
`RFC 1: Roadmap for Shapely 2.0 <https://github.com/shapely/shapely-rfc/pull/1>`__.


Refactor of the internals
^^^^^^^^^^^^^^^^^^^^^^^^^

Before 2.0, Shapely wrapped the underlying GEOS C++ library using
``ctypes``. While being a low-barrier way to wrap a C library, this runtime
linking also entails overhead and robustness issues.
With 2.0, the internals of Shapely have been refactored to remove the usage
of ``ctypes``, and instead expose the GEOS functionality through a Python C
extension module.

The pointer to the actual GEOS Geometry object is stored in a lightweight
`Python extension type <https://docs.python.org/3/extending/newtypes_tutorial.html>`__.
A single `Geometry` Python extension type is defined in C wrapping a
`GEOSGeometry` pointer. This extension type is further subclassed in Python
to provide the geometry type-specific classes from Shapely (Point,
LineString, Polygon, etc).
By using an extension type defined in C, the GEOS pointer is accessible from
C without Python overhead as a static attribute of the Python object (an
attribute of the C struct that makes up a Python object). This allows writing
vectorized functions on that level, avoiding Python overhead while looping
over the array (see next section).


Top-level vectorized (element-wise) functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Shapely 2.0 provides functionality to work arrays of geometry objects and
exposes all GEOS operations as vectorized functions with a familiar NumPy
interface. This gives a nice user interface and will provide a substantial
performance improvement when working with arrays of geometries.
This adds a required dependency on numpy.

Before the 2.0 release, Shapely only provided an interface for scalar
geometry objects. When organizing many Shapely geometries in arrays, users
have to loop over the array to call the scalar methods/properties. This is
both more verbose to use and has a large overhead limiting the performance of
such applications.

With the 2.0 release, Shapely introduces new vectorized functions operating
on numpy arrays. Those functions are implemented as
`NumPy *universal functions* <https://numpy.org/doc/stable/reference/ufuncs.html>`__
(or ufunc for short). A universal function is a function that operates on
n-dimensional arrays in an element-by-element fashion, supporting array
broadcasting. The for-loops that are involved are fully implemented in C
diminishing the overhead of the Python interpreter.

This adds a required dependency on numpy.

Illustrating this functionality using a small array of points and a single
polygon::

  >>> import shapely
  >>> from shapely import Point
  >>> import numpy as np
  >>> geoms = np.array([Point(0, 0), Point(1, 1), Point(2, 2)])
  >>> polygon = shapely.box(0, 0, 2, 2)

Instead of using a manual for loop, as one would do in previous versions of
Shapely::

  >>> [polygon.contains(point) for point in geoms]
  [False,  True, False]

we can now compute whether the points are contained in the polygon directly
with one function call::

  >>> shapely.contains(polygon, geoms)
  array([False,  True, False])

Apart from the nicer user interface (no need to manually loop through the
geometries), this also provides a considerable speedup. Depending on the
operation, this can give a performance increase with factors of 4x to 100x
(the high factors are obtained for lightweight GEOS operations such as
contains in which case the Python overhead is the dominating factor). See
https://caspervdw.github.io/Introducing-Pygeos/ for more detailed examples.

Those new vectorized functions are available in the top-level ``shapely``
namespace. All the familiar geospatial methods and attributes from the
geometry classes now have an equivalent as top-level function (with some
small name deviations, such as the ``.wkt`` attribute being available as a
``to_wkt()`` function). In addition, also some methods from submodules (for
example, several functions from the ``shapely.ops`` submodule such as
``polygonize()``) are also made available in a vectorized version as
top-level function.

A full list of functions can be found in the API docs. TODO add link

* Vectorized constructor functions
* Optionally output to a user-specified array (``out`` keyword argument) when constructing
  geometries from ``indices`` (#380).
* Enable bulk construction of geometries with different number of coordinates
  by optionally taking index arrays in all creation functions (#230, #322, #326, #346).


API changes (deprecated in 1.8)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Shapely 1.8 release included several deprecation warnings about API
changes that would happen in Shapely 2.0 and that can be fixed in your code.
See :ref:`migration` for more details on how to update your code.

It is hightly recommended to first upgrade to Shapely 1.8 and resolve all deprecation
warnings before upgrading to Shapely 2.0.

Summary of changes:

* Geometries are now immutable and hashable.
* Multi-part geometries such as MultiPolygon no longer behave as "sequences".
  This means that they no longer have a length, are not iterable and
  indexable anymore. Use ``.geoms`` attribute instead to access the parts.
* Geometries no longer directly implement the numpy array interface. To
  convert to an array of coordinates, use ``.coords`` instead
  (``np.asarray(geom.coords)``).
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
  the ``_geom`` attribute is still present (#1417).

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

In addition, this also opens up possibilities for multithreading (release the
GIL during GEOS operations for better performance). See
[pygeos/pygeos#113](https://github.com/pygeos/pygeos/pull/113) for experiments
on this.

Shapely functions generally support multithreading by releasing the Global
Interpreter Lock (GIL) during execution. Normally in Python, the GIL prevents
multiple threads from computing at the same time. Shapely functions
internally release this constraint so that the heavy lifting done by GEOS can
be done in parallel, from a single Python process.

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
