Shapely
=======

Shapely is a Python package for manipulation and analysis of 2D geospatial
geometries. It is based on GEOS (http://geos.refractions.net).  Shapely 1.0 is
not concerned with data formats or coordinate reference systems.
Responsibility for reading and writing data and projecting coordinates is left
to other packages like WorldMill_ and pyproj_. For more information, see:

* Shapely wiki_
* Shapely manual_

Shapely requires Python 2.4+. (I've also begun to port it to Python 3.0:
http://zcologia.com/news/564/shapely-for-python-3-0/.)

.. note::
   We've switched to Windows GEOS DLLs based on MinGW in versions >= 1.0.6.
   Please contact us if you experience difficulties.

See also CHANGES.txt_ and HISTORY.txt_.

.. _CHANGES.txt: http://trac.gispython.org/projects/PCL/browser/Shapely/trunk/CHANGES.txt
.. _HISTORY.txt: http://trac.gispython.org/projects/PCL/browser/Shapely/trunk/HISTORY.txt
.. _WorldMill: http://pypi.python.org/pypi/WorldMill
.. _pyproj: http://pypi.python.org/pypi/pyproj


Dependencies
------------

* libgeos_c (2.2.3 or 3.0.0+)
* Python ctypes_ (standard in Python 2.5+)

.. _ctypes: http://pypi.python.org/pypi/ctypes/


Installation
------------

Windows users should use the executable installer, which contains the required
GEOS DLL. Other users should acquire libgeos_c by any means, make sure that it
is on the system library path, and install from the Python package index::

  $ sudo easy_install Shapely

with the setup script::

  $ sudo python setup.py install

or by using the development buildout on Linux, which also provides libgeos_c::

  $ svn co http://svn.gispython.org/svn/gispy/buildout/shapely.buildout/trunk shapely.buildout
  $ cd shapely.buildout
  $ python bootstrap.py
  $ ./bin/buildout


Usage
-----

To buffer a point::

  >>> from shapely.geometry import Point
  >>> point = Point(-106.0, 40.0) # longitude, latitude
  >>> point.buffer(10.0)
  <shapely.geometry.polygon.Polygon object at ...>

See the manual_ for comprehensive examples of usage. See also Operations.txt
and Predicates.txt under tests/ for more examples of the spatial operations and
predicates provided by Shapely. See also Point.txt, LineString.txt, etc for
examples of the geometry APIs.


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
  >>> line.wkt
  'LINESTRING (1.0000000000000000 2.0000000000000000, 3.0000000000000000 4.0000000000000000)'


Python Geo Interface
--------------------

Any object that provides the Python geo interface can be adapted to a Shapely
geometry with the asShape factory::

  >>> d = {"type": "Point", "coordinates": (0.0, 0.0)}
  >>> from shapely.geometry import asShape
  >>> shape = asShape(d)
  >>> shape.geom_type
  'Point'
  >>> tuple(shape.coords)
  ((0.0, 0.0),)

  >>> class GeoThing(object):
  ...     def __init__(self, d):
  ...         self.__geo_interface__ = d
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
zope.testing. runalldoctests.py does not. Perhaps the easiest way to run the 
tests is::

  $ python setup.py test


Support
-------

For current information about this project, see the wiki_.

.. _wiki: http://trac.gispython.org/projects/PCL/wiki/Shapely
.. _manual: http://gispython.org/shapely/manual.html

If you have questions, please consider joining our community list:

http://trac.gispython.org/projects/PCL/wiki/CommunityList


Credits
-------

* Sean Gillies (Pleiades)
* Howard Butler (Hobu, Inc.)
* Kai Lautaportti (Hexagon IT)
* Fr |eaigue| d |eaigue| ric Junod (Camptocamp SA)
* Eric Lemoine (Camptocamp SA)
* Justin Bronn (GeoDjango) for ctypes inspiration

.. |eaigue| unicode:: U+00E9
   :trim:

Major portions of this work were supported by a grant (to Pleiades) from the
U.S.  National Endowment for the Humanities (http://www.neh.gov).
