Shapely
=======

Dependencies
------------

GeoJSON


Testing
-------

Several of the modules have docstring doctests::

  $ cd shapely
  $ python point.py
  $ python factory.py

The point tests show a problem that I'm having with setting coordinate values.

**Update**: it's a known but not understood or documented bug in the GEOS C
API. See 

http://svn.refractions.net/geos/trunk/tests/unit/capi/GEOSCoordSeqTest.cpp
