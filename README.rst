======
PyGEOS
======

This is a C/Python library that wraps geometry functions in GEOS in numpy ufuncs.
This project is still in a mock-up phase: the API will most likely change.

The Geometry object
-------------------

GEOS geometry objects are stored in a Python extension type `pygeos.BaseGeometry`,
that keeps the python interpreter out of the numpy ufunc inner loop. This
 object calls the GEOS `destroy` function to deallocate memory
just before the wrapping `Geometry` object is deallocated.

Ufuncs only act on these `pygeos.BaseGeometry` objects.
Construct these as follows::

.. code:: python

  >>> from pygeos import BaseGeometry
  >>> from shapely.geometry import Point

  >>> point = BaseGeometry(Point(i, j))

Or simply:

.. code:: python

  >>> from pygeos import Point

  >>> point = Point(5.2, 52.1)

Examples
--------

Compare an grid of points with a polygon:

.. code:: python

  >>> from pygeos import contains, box, Point

  >>> points = [[Point(i, j) for i in range(4)] for j in range(4)]
  >>> polygon = box(0, 0, 2, 2)

  >>> contains(polygon, points)

    array([[False, False, False, False],
           [False,  True, False, False],
           [False, False, False, False],
           [False, False, False, False]])


Compute the area of all possible intersections of two lists of polygons:

.. code:: python

  >>> from pygeos import box, area, intersection

  >>> polygons_x = [[box(0 + i, 0, 10 + i, 10) for i in range(5)]]
  >>> polygons_y = [[box(0, 0 + j, 10, 10 + j)] for j in range(5)]

  >>> area(intersection(polygons_x, polygons_y))

  array([[100.,  90.,  80.,  70.,  60.],
       [ 90.,  81.,  72.,  63.,  54.],
       [ 80.,  72.,  64.,  56.,  48.],
       [ 70.,  63.,  56.,  49.,  42.],
       [ 60.,  54.,  48.,  42.,  36.]])

Installation
------------

Pygeos uses shapely's installing scripts. If you have libgeos at a standard
location, the following should work::

    $ pip install pygeos


Installation for developers
---------------------------

Clone the package::

    $ git clone https://github.com/caspervdw/pygeos.git

Install it using `pip`::

    $ pip install -e .

Run the unittests::

    $ python -m pytest pygeos/test.py

References
----------

 - GEOS: http://trac.osgeo.org/geos
 - Shapely: https://shapely.readthedocs.io/en/latest/
 - Numpy ufuncs: https://docs.scipy.org/doc/numpy/reference/ufuncs.html
 - Joris van den Bossche's blogpost: https://jorisvandenbossche.github.io/blog/2017/09/19/geopandas-cython/
 - Matthew Rocklin's blogpost: http://matthewrocklin.com/blog/work/2017/09/21/accelerating-geopandas-1


Copyright & License
-------------------

Copyright (c) 2019, Casper van der Wel. BSD 3-Clause license.
