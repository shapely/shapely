.. pygeos documentation master file, created by
   sphinx-quickstart on Mon Jul 22 11:02:13 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the PyGEOS documentation
===================================

PyGEOS is a C/Python library with vectorized geometry functions. The geometry
operations are done in the open-source geometry library GEOS. PyGEOS wraps
these operations in NumPy ufuncs providing a performance improvement when
operating on arrays of geometries.

Why ufuncs?
-----------

A universal function (or ufunc for short) is a function that operates on
n-dimensional arrays in an element-by-element fashion, supporting array
broadcasting. The for-loops that are involved are fully implemented in C
diminishing the overhead of the Python interpreter.


The Geometry object
-------------------

The `pygeos.Geometry` object is a container of the actual GEOSGeometry object.
A C pointer to this object is stored in a static attribute of the `Geometry`
object. This keeps the python interpreter out of the ufunc inner loop. The
Geometry object keeps track of the underlying GEOSGeometry and
allows the python garbage collector to free memory when it is not
used anymore.

`Geometry` objects are immutable. Construct them as follows:

.. code:: python

  >>> from pygeos import Geometry

  >>> geometry = Geometry("POINT (5.2 52.1)")

Or using one of the provided (vectorized) functions:

.. code:: python

  >>> from pygeos import points

  >>> point = points(5.2, 52.1)


API Reference
=============

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   creation
   constructive
   coordinates
   geometry
   io
   linear
   measurement
   predicates
   set_operations
   strtree
   changelog


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
