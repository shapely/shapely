======
PyGEOS
======

.. Documentation at RTD — https://readthedocs.org

.. image:: https://readthedocs.org/projects/pygeos/badge/?version=latest
	:alt: Documentation Status
	:target: https://pygeos.readthedocs.io/en/latest/?badge=latest

.. Github Actions status — https://github.com/pygeos/pygeos/actions

.. image:: https://github.com/pygeos/pygeos/workflows/Conda/badge.svg
	:alt: Github Actions status
	:target: https://github.com/pygeos/pygeos/actions?query=workflow%3AConda

.. Appveyor CI status — https://ci.appveyor.com

.. image:: https://ci.appveyor.com/api/projects/status/jw48gpd88f188av6?svg=true
	:alt: Appveyor CI status
	:target: https://ci.appveyor.com/project/caspervdw/pygeos-3e5cu

.. PyPI

.. image:: https://badge.fury.io/py/pygeos.svg
	:alt: PyPI
	:target: https://badge.fury.io/py/pygeos

.. Anaconda

.. image:: https://anaconda.org/conda-forge/pygeos/badges/version.svg
  :alt: Anaconda

.. Zenodo

.. image:: https://zenodo.org/badge/191151963.svg
  :alt: Zenodo 
  :target: https://zenodo.org/badge/latestdoi/191151963


PyGEOS is a C/Python library with vectorized geometry functions. The geometry
operations are done in the open-source geometry library GEOS. PyGEOS wraps
these operations in NumPy ufuncs providing a performance improvement when
operating on arrays of geometries.

Note: PyGEOS is a very young package. While the available functionality should
be stable and working correctly, it's still possible that APIs change in upcoming
releases. But we would love for you to try it out, give feedback or contribute!

What is a ufunc?
----------------

A universal function (or ufunc for short) is a function that operates on
n-dimensional arrays in an element-by-element fashion, supporting array
broadcasting. The for-loops that are involved are fully implemented in C
diminishing the overhead of the Python interpreter.

Multithreading
--------------

PyGEOS functions support multithreading. More specifically, the Global
Interpreter Lock (GIL) is released during function execution. Normally in Python, the
GIL prevents multiple threads from computing at the same time. PyGEOS functions
internally releases this constraint so that the heavy lifting done by GEOS can be
done in parallel, from a single Python process.

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

See the documentation for more: https://pygeos.readthedocs.io


Relationship to Shapely
-----------------------

Both Shapely and PyGEOS are exposing the functionality of the GEOS C++ library
to Python. While Shapely only deals with single geometries, PyGEOS provides
vectorized functions to work with arrays of geometries, giving better
performance and convenience for such usecases.

There is active discussion and work toward integrating PyGEOS into Shapely:

* latest proposal: https://github.com/shapely/shapely-rfc/pull/1
* prior discussion: https://github.com/Toblerity/Shapely/issues/782

For now PyGEOS is developed as a separate project.

References
----------

- GEOS: http://trac.osgeo.org/geos
- Shapely: https://shapely.readthedocs.io/en/latest/
- Numpy ufuncs: https://docs.scipy.org/doc/numpy/reference/ufuncs.html
- Joris van den Bossche's blogpost: https://jorisvandenbossche.github.io/blog/2017/09/19/geopandas-cython/
- Matthew Rocklin's blogpost: http://matthewrocklin.com/blog/work/2017/09/21/accelerating-geopandas-1


Copyright & License
-------------------

PyGEOS is licensed under BSD 3-Clause license. Copyright (c) 2019, Casper van der Wel.
GEOS is available under the terms of ​GNU Lesser General Public License (LGPL) 2.1 at https://trac.osgeo.org/geos.
