======
PyGEOS
======

.. Travis CI status — https://travis-ci.org

.. image:: https://travis-ci.org/caspervdw/pygeos.svg?branch=master
	:alt: Travis CI status
	:target: https://travis-ci.org/caspervdw/pygeos

.. Appveyor CI status — https://ci.appveyor.com

.. image:: https://ci.appveyor.com/api/projects/status/yx6nmovs0wq8eg9n/branch/master?svg=true
	:alt: Appveyor CI status
	:target: https://ci.appveyor.com/project/caspervdw/pygeos

This is a C/Python library that wraps geometry functions in GEOS in numpy ufuncs.
This project is still in a mock-up phase: the API will most likely change.


Why ufuncs?
-----------

A universal function (or ufunc for short) is a function that operates on
n-dimensional arrays in an element-by-element fashion, supporting array
broadcasting. The for-loops that are involved are fully implmemented in C,
diminishing the overhead of the python interpreter.

Pygeos aims to expose the geometry functions from GEOS into python to provide
a fast and flexible means to work with large sets of geometries from python.


The Geometry object
-------------------

GEOS geometry objects are stored in a static attribute of the Python extension
type `pygeos.Geometry`. This keeps the python interpreter out of the ufunc
inner loop. The Geometry object keeps track of the underlying geometry and
allows the python garbage collector to free memory when the geometry is not
used anymore.

`Geometry` objects are immutable. Construct them as follows:

.. code:: python

  >>> from pygeos import Geometry

  >>> geometry = Geometry.from_wkt("POINT (5.2 52.1)")

Or simply:

.. code:: python

  >>> from pygeos import points

  >>> point = points(5.2, 52.1)

Examples
--------

Compare an grid of points with a polygon:

.. code:: python

  >>> geoms = points(*np.indices((4, 4)))
  >>> polygon = box(0, 0, 2, 2)

  >>> contains(polygon, geoms)

    array([[False, False, False, False],
           [False,  True, False, False],
           [False, False, False, False],
           [False, False, False, False]])


Compute the area of all possible intersections of two lists of polygons:

.. code:: python

  >>> from pygeos import box, area, intersection

  >>> polygons_x = box(range(5), 0, range(10, 15), 10)
  >>> polygons_y = box(0, range(5), 10, range(10, 15))

  >>> area(intersection(polygons_x[:, np.newaxis], polygons_y[np.newaxis, :]))

  array([[100.,  90.,  80.,  70.,  60.],
       [ 90.,  81.,  72.,  63.,  54.],
       [ 80.,  72.,  64.,  56.,  48.],
       [ 70.,  63.,  56.,  49.,  42.],
       [ 60.,  54.,  48.,  42.,  36.]])

Installation using conda
------------------------

Pygeos requires the presence of NumPy and GEOS >= 3.5. It is recommended to obtain
these using Anaconda::

    $ conda install numpy geos

On Linux / OSX::

    $ export GEOS_INCLUDE_PATH=$CONDA_PREFIX/Library/include
    $ export GEOS_LIBRARY_PATH=$CONDA_PREFIX/Library/lib
    $ pip install pygeos

On windows (assuming you are in a Visual C++ shell)::

    $ set GEOS_INCLUDE_PATH=%CONDA_PREFIX%\Library\include
    $ set GEOS_LIBRARY_PATH=%CONDA_PREFIX%\Library\lib
    $ pip install pygeos


Installation using system GEOS
------------------------------

On Linux::

    $ sudo apt install libgeos-dev

On OSX::

    $ brew install geos

Make sure `geos-config` is available from you shell; append PATH if necessary::

    $ export PATH=$PATH:/path/to/dir/having/geos-config
    $ pip install pygeos


Installation for developers
---------------------------

Clone the package::

    $ git clone https://github.com/caspervdw/pygeos.git

Install it using `pip`::

    $ pip install -e .[test]

Run the unittests::

    $ pytest

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
