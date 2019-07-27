.. pygeos documentation master file, created by
   sphinx-quickstart on Mon Jul 22 11:02:13 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pygeos's documentation
=================================

Pygeos is a C/Python library that implements vectorized versions of the
geometry functions from GEOS. By wrapping the GEOS operations in Numpy
"universal functions", pygeos provides a significant performance improvement
when working with large sets of geometries.

Why ufuncs?
-----------

A universal function (or ufunc for short) is a function that operates on
n-dimensional arrays in an element-by-element fashion, supporting array
broadcasting. The for-loops that are involved are fully implmemented in C
diminishing the overhead of the python interpreter.

.. note::
   This documentation is still work in progress: only a part of the API has
   been documented.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   predicates


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


