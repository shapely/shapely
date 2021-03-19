Frequently asked questions and answers
======================================

I installed shapely in a conda environment using pip. Why doesn't it work?
--------------------------------------------------------------------------

Shapely versions < 2.0 load a GEOS shared library using ctypes. It's not uncommon for users to have
multiple copies of GEOS libs on their system. Loading the correct one is complicated and shapely has
a number of platform-dependent GEOS library loading bugs. The project has particularly poor support
for finding the correct GEOS library for a shapely package installed from PyPI *into* a conda
environment. We recommend that conda users always get shapely from conda-forge.

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
