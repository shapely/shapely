


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
changes that would happen in Shapely 2.0 and that can be fixed in your code
(making it compatible with both <=1.8 and >=2.0). See :ref:`migration` for
more details on how to update your code.

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

Some new deprecations have been introduced in Shapely 2.0:

* Directly calling the base class ``BaseGeometry()`` constructor, as well as
  the ``EmptyGeometry()`` constructor, are deprecated and will raise an error
  in the future. To create an empty geometry, use one of the subclasses
  instead, for example ``GeometryCollection()`` (#1022).
* The ``shapely.speedups`` module (the ``enable`` and ``disable`` functions)
  is deprecated and will be removed in the future. The module has no longer
  any affect in Shapely >=2.0.


Breaking API changes
^^^^^^^^^^^^^^^^^^^^

Some additional backwards incompatible API changes were included in Shapely
2.0 that were not yet deprecated in Shapely 1.8:

* The ``.bounds`` of an empty geometry has changed from an empty tuple to a
  tuple of NaNs (#1023).
* The default of the ``preserve_topology`` keyword of ``simplify()`` changed
  to True (#1392).
* A ``GeometryCollection`` that consists of all empty sub-geometries now
  returns those empty geometries in ``.geoms`` (instead of returning an empty
  list) (#1420).
* The unused ``shape_factory()`` method and ``HeterogeneousGeometrySequence``
  class are removed (#1421).
* The undocumted ``__geom__`` attribute is removed. To access the raw GEOS pointer,
  the ``_geom`` attribute is still present (#1417).
* The ``logging`` functionality has been removed. All errors messages from
  GEOS are now bubbled up as Python exceptions (#998).
* Several custom exception classes defined in ``shapely.errors`` (that are no
  longer used internally) have been removed. Errors from GEOS are now raised
  as ``GEOSException`` (#1306).

In addition, the ``STRtree`` interface was changed, see the section
:ref:`below <changelog-2-strtree>``for more details.

New features
^^^^^^^^^^^^

The Geometry subclasses are available in the top-level namespace
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Following the new vectorized functions in the top-level ``shapely``
namespace, the Geometry subclasses (``Point``, ``LineString``, ``Polygon``,
etc) are now available in the top-level namespace as well (#1330). Thus it is
no longer needed to import those from the ``shapely.geometry`` submodule.

The following::

  from shapely.geometry import Point

can be replaced with::

  from shapely import Point

  # or
  import shapely
  shapely.Point(...)

Note: for backwards compatibility (and being able to write code that works
for both <=1.8 and >2.0), those classes still remain accessible from the
``shapely.geometry`` submodule as well.


More informative repr with (truncated) WKT
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The repr (``__repr__``) of Geometry objects has been simplified and improved
to include a descriptive Well-Known-Text (WKT) formatting. Instead of showing
the class name and id::

  >>> Point(0, 0)
  <shapely.geometry.point.Point at 0x7f0b711f1310>

we now get::

  >>> Point(0, 0)
  <POINT (0 0)>

For large geometries with many coordinates, the WKT string gets truncated at
80 characters.


Support for fixed precision model for geometries and in overlay functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GEOS 3.9.0 overhauled the overlay operations (union, intersection,
(symmetric) difference): a complete rewrite, dubbed "OverlayNG", provides a
more robust implementation (no more TopologyExceptions even on valid input),
the ability to specify the output precision model, and significant
performance optimizations. When installing Shapely with GEOS >= 3.9 (which is
the case for PyPI wheels and conda-forge packages), you automatically get
those improvements already (also for previous versions of Shapely) when using
the overlay operations.

An additional improvement in Shapely 2.0 is that the ability to specify the
precision model is now exposed in the Python API:

* The ``set_precision()`` function can be used to conform a geometry to a
  certain grid size (may round and reduce coordinates), and this will then
  also be used by subsequent overlay methods. A ``get_precision()`` function
  is also available to inspect the precision model of geometries.
* The ``grid_size`` keyword in the overlay methods can also be used to
  specify the precision model of the output geometry (without first
  conforming the input geometries).


Releasing the GIL for multithreaded applications
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Shapely itself is not multithreaded, but its functions generally allow for
multithreading by releasing the Global Interpreter Lock (GIL) during
execution. Normally in Python, the GIL prevents multiple threads from
computing at the same time. Shapely functions internally release this
constraint so that the heavy lifting done by GEOS can be done in parallel,
from a single Python process.


.. _changelog-2-strtree:

STRtree API changes and improvements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The biggest change in the ``STRtree`` interface is that all operations now
return indices of the input tree or query geometries, instead of the
geometries itself. These indices can be used to index into anything
associated with the input geometries, including the input geometries
themselves, or custom items stored in another object of the same length as
the geometries.

In addition, several significant improvements in the ``STRtree`` are included
in Shapely 2.0:

* Directly include predicate evaluation in ``STRtree.query()`` by specifying
  the ``predicate`` keyword. If a predicate is provided, the potentially
  intersecting tree geometries are further filtered to those that meet the
  predicate (using prepared geometries under the hood for efficiency).
* Query multiple input geometries (spatial join style) with
  ``STRtree.query()`` by passing an array of geometries. In this case, the
  return value is a 2D array with shape (2, n) where the subarrays correspond
  to the indices of the input geometries and indices of the tree geometries
  associated with each.
* A new ``STRtree.query_nearest()`` method was added, returning the index of
  the nearest geometries in the tree for each input geometry. Compared to
  ``STRtree.nearest()``, which only returns the index of a single nearest
  geometry for each input geometry, this new methods allows for:

  * returning all equidistant nearest geometries,
  * excluding nearest geometries that are equal to the input,
  * specifying an ``max_distance`` to limit the search radius potentially
    increasing the performance,
  * optionally returning the distance.

* Fixed ``STRtree`` creation to allow querying the tree in a multi-threaded
  context.

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
``get_dimensions``, ``get_coordinate_dimension``, ``get_srid``,
``get_num_points``, ``get_num_interior_rings``, ``get_num_geometries``,
``get_num_coordinates``, ``get_precision``.

Several functions are added to extract parts: ``get_geometry`` to get the nth
geometry from a GeometryCollection or Multi-part geometry,
``get_exterior_ring`` and ``get_interior_ring`` to get one of the rings of a
Polygon, ``get_point`` to get the nth point of a linestring or linearring,
and ``get_x``, ``get_y`` and ``get_z`` to get the x/y/z coordinate of a
Point.

In addition, methods to extract all parts or coordinates at once were added:

* The ``get_parts`` function can be used to get individual parts of an array of multipart
  geometries.
* The ``get_rings`` function, similar as ``get_parts`` but specifically
  to extract the rings of Polygon geometries.
* The ``get_coordinates`` function to get all coordinates from a geometry or
  array of goemetries as an array of floats.

Each of those three functions has an optional ``return_index`` keyword, which
allows to also return the indexes of the original geometries in the source
array.


Prepared geometries
~~~~~~~~~~~~~~~~~~~

Prepared geometries are now no longer separate objects, but geometry objects itself
can be prepared (this makes the ``shapely.prepared`` module superfluous).

The ``prepare()`` function generates a GEOS prepared geometry which is stored
on the Geometry object itself. All binary predicates (except ``equals``) will
make use of this, if the input geometry has been prepared. Helper functions
``destroy_prepared`` and ``is_prepared`` are also available.


GeoJSON IO
~~~~~~~~~~

* Added GeoJSON input/output capabilities (``shapely.from_geojson``,
  ``shapely.to_geojson``) for GEOS >= 3.10 (#413).

Other improvements
~~~~~~~~~~~~~~~~~~

* Added ``shapely.force_2d`` and ``shapely.force_3d`` to change the dimensionality of
  the coordinates in a geometry.
* Addition of a ``total_bounds()`` function.
* Added ``shapely.empty`` to create a geometry array pre-filled with None or
  with empty geometries.
* Performance improvement in constructing LineStrings or LinearRings from
  numpy arrays for GEOS >= 3.10.
* Updated ``box`` ufunc to use internal C function for creating polygon
  (about 2x faster) and added ``ccw`` parameter to create polygon in
  counterclockwise (default) or clockwise direction.
* Start of a benchmarking suite using ASV.

Utilities

* Added ``shapely.testing.assert_geometries_equal``.


Bug fixes
~~~~~~~~~

* Fixed several corner cases in WKT and WKB serialization for varying GEOS
  versions, including:

  * Fixed the WKT serialization of single part 3D empty geometries to
    correctly include "Z" (for GEOS >= 3.9.0).
  * Handle empty points in WKB serialization by conversion to
    ``POINT (nan, nan)`` consistently for all GEOS versions (GEOS started
    doing this for >= 3.9.0).


**Acknowledgments**

Thanks to everyone who contributed to this release!
People with a "+" by their names contributed a patch for the first time.

* Adam J. Stewart +
* Alan D. Snow +
* Brendan Ward +
* Casper van der Wel +
* James Myatt +
* Joris Van den Bossche
* Keith Jenkins +
* Kian Meng Ang +
* Krishna Chaitanya +
* Martin Fleischmann +
* Martin Lackner +
* Mike Taves
* Tanguy Ophoff +
* Tom Clancy
* Sean Gillies
* gpapadok +
* mattijn +
* odidev +
