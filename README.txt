=======
Shapely
=======

.. image:: http://farm3.static.flickr.com/2738/4511827859_b5822043b7_o_d.png
   :width: 800
   :height: 400

Shapely is a BSD-licensed Python package for manipulation and analysis of
planar geometric objects. It is not concerned with data formats or coordinate
systems.  It is based on the widely deployed GEOS_ (the engine of PostGIS_) and
JTS_ (from which GEOS is ported) libraries. This C dependency is traded for the
ability to execute with blazing speed.

In a nutshell: Shapely lets you do PostGIS-ish stuff outside the context of a
database using idiomatic Python. For more details, see:

* Shapely wiki_
* Shapely manual_
* Shapely `example apps`_

Dependencies
------------

Shapely 1.2 depends on:

* Python >=2.5,<3
* libgeos_c >=3.1 (3.0 and below have not been tested, YMMV)

Installation
------------

Windows users should use the executable installer, which contains the required
GEOS DLL. Other users should acquire libgeos_c by any means, make sure that it
is on the system library path, and install from the Python package index::

  $ pip install Shapely

or from a source distribution with the setup script::

  $ python setup.py install

Usage
-----

Here is the canonical example of building an approximately circular patch by
buffering a point::

  >>> from shapely.geometry import Point
  >>> patch = Point(0.0, 0.0).buffer(10.0)
  >>> patch
  <shapely.geometry.polygon.Polygon object at 0x...>
  >>> patch.area
  313.65484905459385

See the manual_ for comprehensive usage snippets and the dissolve.py and
intersect.py `example apps`_.

Integration 
-----------

Shapely does not read or write data files, but it can serialize and deserialize
using several well known formats and protocols. The shapely.wkb and shapely.wkt
modules provide dumpers and loaders inspired by Python's pickle module.::

  >>> from shapely.wkt import dumps, loads
  >>> dumps(loads('POINT (0 0)'))
  'POINT (0.0000000000000000 0.0000000000000000)'

All linear objects, such as the rings of a polygon (like ``patch`` above),
provide the Numpy array interface.::

  >>> from numpy import asarray
  >>> ag = asarray(patch.exterior)
  >>> ag
  array([[  1.00000000e+01,   0.00000000e+00],
         [  9.95184727e+00,  -9.80171403e-01],
         [  9.80785280e+00,  -1.95090322e+00],
         ...
         [  1.00000000e+01,   0.00000000e+00]])

That yields a numpy array of [x, y] arrays. This is not always exactly what one
wants for plotting shapes with Matplotlib, so Shapely 1.2 adds a `xy` property
for getting separate arrays of coordinate x and y values.::

  >>> x, y = patch.exterior.xy
  >>> ax = asarray(x)
  >>> ax
  array([  1.00000000e+01,   9.95184727e+00,   9.80785280e+00,  ...])

Numpy arrays can also be adapted to Shapely linestrings::

  >>> from shapely.geometry import asLineString
  >>> asLineString(ag).length
  62.806623139095073
  >>> asLineString(ag).wkt
  'LINESTRING (10.0000000000000000 0.0000000000000000, ...)'

Testing
-------

Shapely uses a Zope-stye suite of unittests and doctests, excercised via
setup.py.::

  $ python setup.py test

Nosetests won't run the tests properly; Zope doctest suites are not currently
supported well by nose.

Support
-------

For current information about this project, see the wiki_.

If you have questions, please consider joining our community list:

http://trac.gispython.org/projects/PCL/wiki/CommunityList

Credits
-------

Shapely is written by:

* Sean Gillies
* Aron Bierbaum
* Kai Lautaportti

Patches contributed by:

* Howard Butler
* Fr |eaigue| d |eaigue| ric Junod
* Eric Lemoine
* Jonathan Tartley
* Kristian Thy
* Oliver Tonnhofer

Additional help from:

* Justin Bronn (GeoDjango) for ctypes inspiration
* Martin Davis (JTS)
* Jaakko Salli for the Windows distributions
* Sandro Santilli, Mateusz Loskot, Paul Ramsey, et al (GEOS Project)

Major portions of this work were supported by a grant (for Pleiades_) from the
U.S. National Endowment for the Humanities (http://www.neh.gov).

.. _JTS: http://www.vividsolutions.com/jts/jtshome.htm
.. _PostGIS: http://postgis.org
.. _GEOS: http://trac.osgeo.org/geos/
.. _example apps: http://trac.gispython.org/lab/wiki/Examples
.. _wiki: http://trac.gispython.org/lab/wiki/Shapely
.. _manual: http://gispython.org/shapely/docs/1.2
.. |eaigue| unicode:: U+00E9
   :trim:
.. _Pleiades: http://pleiades.stoa.org
