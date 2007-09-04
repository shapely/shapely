Shapely
=======

Shapely is a Python package for programming with geospatial geometries. It is
based on GEOS (http://geos.refractions.net). Shapely 1.0 is ignorant about
coordinate and reference systems. Projection responsibility is left to specific
applications. For more information, see

http://trac.gispython.org/projects/PCL/wiki/ShapeLy

Dependencies
------------

* libgeos_c (2.2.3)
* ctypes

Installation
------------

Shapely can be installed from the Python package index::

  $ sudo easy_install Shapely

with the setup script::

  $ sudo python setup.py install

or by using the development buildout, which also provides libgeos_c::

 $ svn co http://svn.gispython.org/svn/gispy/buildout/shapely.buildout/trunk shapely.buildout
 $ cd shapely.buildout
 $ python bootstrap.py
 $ ./bin/buildout

Usage
-----

Buffer a point::

  >>> from shapely.geometry import Point
  >>> point = Point(-106.0, 40.0) # longitude, latitude
  >>> point.buffer(10.0)
  <shapely.geometry.polygon.Polygon object at ...>

See Operations.txt and Predicates.txt under tests/ for more examples of the
spatial operations and predicates provided by Shapely. Also see Point.txt,
LineString.txt, etc for examples of the geometry APIs.

Numpy integration
-----------------

All Shapely geometry instances provide the Numpy array interface::

  >>> from numpy import asarray
  >>> a = asarray(point)
  >>> a.size
  3
  >>> a.shape
  (2,)

Numpy arrays can also be adapted to Shapely points and linestrings::

  >>> from shapely.geometry import asLineString
  >>> a = array([[1.0, 2.0], [3.0, 4.0]])
  >>> line = asLineString(a)
  >>> la.wkt
  'LINESTRING (1.0000000000000000 2.0000000000000000, 3.0000000000000000 4.0000000000000000)'

Python Geo Interface
--------------------

Any object that provides the Python geo interface can be adapted to a Shapely
geometry with the asShape factory::

  >>> d = {"type": "Point", "coordinates": (0.0, 0.0)}
  >>> shape = asShape(d)
  >>> shape.geom_type
  'Point'
  >>> tuple(shape.coords)
  ((0.0, 0.0),)

  >>> class GeoThing(object):
  ...     def __init__(self, d):
  ...         self.__geo_interface__ = d

  >>> shape = None
  >>> thing = GeoThing({"type": "Point", "coordinates": (0.0, 0.0)})
  >>> shape = asShape(thing)
  >>> shape.geom_type
  'Point'
  >>> tuple(shape.coords)
  ((0.0, 0.0),)

See http://trac.gispython.org/projects/PCL/wiki/PythonGeoInterface for more
details on the interface.

Testing
-------

Several of the modules have docstring doctests::

  $ cd shapely
  $ python point.py

There are also two test runners under tests/. test_doctests.py requires
zope.testing. runalldoctests.py does not.

