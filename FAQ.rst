Frequently asked questions and answers
======================================

Are there references for the algorithms used by shapely?
--------------------------------------------------------

Generally speaking, shapely's predicates and operations are derived from
methods of the same name from GEOS_ and the `JTS Topology Suite`_.  See the `JTS FAQ`_ for references
describing the JTS algorithms.

I used .buffer() on a geometry with Z coordinates. Where did the Z coordinates go?
----------------------------------------------------------------------------------

The buffer algorithm in GEOS_ is purely two-dimensional and discards any Z coordinates.
This is generally the case for the GEOS algorithms.


.. _GEOS: https://trac.osgeo.org/geos/
.. _JTS Topology Suite: https://locationtech.github.io/jts/
.. _JTS FAQ: https://locationtech.github.io/jts/jts-faq.html#E1
