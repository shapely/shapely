=======
Shapely
=======

.. Documentation at RTD — https://readthedocs.org

.. image:: https://readthedocs.org/projects/shapely/badge/?version=stable
   :alt: Documentation Status
   :target: https://shapely.readthedocs.io/en/stable/

.. Github Actions status — https://github.com/shapely/shapely/actions

.. |github-actions| image:: https://github.com/shapely/shapely/workflows/Tests/badge.svg?branch=main
   :alt: Github Actions status
   :target: https://github.com/shapely/shapely/actions?query=branch%3Amain

.. Travis CI status -- https://travis-ci.com

.. image:: https://travis-ci.com/shapely/shapely.svg?branch=main
   :alt: Travis CI status
   :target: https://travis-ci.com/github/shapely/shapely

.. PyPI

.. image:: https://img.shields.io/pypi/v/shapely.svg
   :alt: PyPI
   :target: https://pypi.org/project/shapely/

.. Anaconda

.. image:: https://img.shields.io/conda/vn/conda-forge/shapely
   :alt: Anaconda
   :target: https://anaconda.org/conda-forge/shapely

.. Coverage

.. |coveralls| image:: https://coveralls.io/repos/github/shapely/shapely/badge.svg?branch=main
   :target: https://coveralls.io/github/shapely/shapely?branch=main

.. Zenodo

.. .. image:: https://zenodo.org/badge/191151963.svg
..   :alt: Zenodo
..   :target: https://zenodo.org/badge/latestdoi/191151963

Manipulation and analysis of geometric objects in the Cartesian plane.

.. image:: https://c2.staticflickr.com/6/5560/31301790086_b3472ea4e9_c.jpg
   :width: 800
   :height: 378

Shapely is a BSD-licensed Python package for manipulation and analysis of
planar geometric objects. It is using the widely deployed open-source
geometry library `GEOS <https://libgeos.org/>`__ (the engine of `PostGIS
<https://postgis.net/>`__, and a port of `JTS <https://locationtech.github.io/jts/>`__).
Shapely wraps the GEOS operations in NumPy ufuncs providing a performance
improvement when operating on arrays of geometries, as well as provides a
rich scalar `Geometry` interface.
Shapely is not concerned with data formats or coordinate systems, but can be
readily integrated with packages that are.

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

Usage
=====

Here is the canonical example of building an approximately circular patch by
buffering a point, using the scalar Geometry interface:

.. code-block:: pycon

    >>> import shapely
    >>> patch = shapely.Point(0.0, 0.0).buffer(10.0)
    >>> patch
    <shapely.Polygon POLYGON ((10 0, 9.952 -0.98, 9.808 -1.951, 9.569 -2.903, 9....>
    >>> patch.area
    313.6548490545941

Using the vectorized ufunc interface, compare a grid of points with a polygon:

.. code:: python

    >>> geoms = shapely.points(*np.indices((4, 4)))
    >>> polygon = shapely.box(0, 0, 2, 2)

    >>> shapely.contains(polygon, geoms)
    array([[False, False, False, False],
           [False,  True, False, False],
           [False, False, False, False],
           [False, False, False, False]])

Compute the area of all possible intersections of two lists of polygons:

.. code:: python

    >>> polygons_x = shapely.box(range(5), 0, range(10, 15), 10)
    >>> polygons_y = shapely.box(0, range(5), 10, range(10, 15))

    >>> from shapely import area, intersection
    >>> area(intersection(polygons_x[:, np.newaxis], polygons_y[np.newaxis, :]))
    array([[100.,  90.,  80.,  70.,  60.],
           [ 90.,  81.,  72.,  63.,  54.],
           [ 80.,  72.,  64.,  56.,  48.],
           [ 70.,  63.,  56.,  49.,  42.],
           [ 60.,  54.,  48.,  42.,  36.]])

See the documentation for more examples and guidance: https://shapely.readthedocs.io

Requirements
============

Shapely 1.8 requires

* Python >=3.6
* GEOS >=3.5

Installing Shapely
==================

Shapely may be installed from a source distribution or one of several kinds
of built distribution.

Built distributions
-------------------

Built distributions are the only option for users who do not have or do not
know how to use their platform's compiler and Python SDK, and a good option for
users who would rather not bother.

Linux, OS X, and Windows users can get Shapely wheels with GEOS included from the
Python Package Index with a recent version of pip (8+):

.. code-block:: console

    $ pip install shapely

Shapely is available via system package management tools like apt, yum, and
Homebrew, and is also provided by popular Python distributions like Canopy and
Anaconda. If you use the Conda package manager to install Shapely, be sure to
use the conda-forge channel.

Windows users have another good installation options: the wheels published at
https://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely. These can be installed
using pip by specifying the entire URL.

Source distributions
--------------------

If you want to build Shapely from source for compatibility with other modules
that depend on GEOS (such as cartopy or osgeo.ogr) or want to use a different
version of GEOS than the one included in the project wheels you should first
install the GEOS library, Cython, and Numpy on your system (using apt, yum,
brew, or other means) and then direct pip to ignore the binary wheels.

.. code-block:: console

    $ pip install shapely --no-binary shapely

If you've installed GEOS to a standard location, the geos-config program will
be used to get compiler and linker options. If geos-config is not on your
executable, it can be specified with a GEOS_CONFIG environment variable, e.g.:

.. code-block:: console

    $ GEOS_CONFIG=/path/to/geos-config pip install shapely

Integration
===========

Shapely does not read or write data files, but it can serialize and deserialize
using several well known formats and protocols. The shapely.wkb and shapely.wkt
modules provide dumpers and loaders inspired by Python's pickle module.

.. code-block:: pycon

    >>> from shapely.wkt import dumps, loads
    >>> dumps(loads('POINT (0 0)'))
    'POINT (0.0000000000000000 0.0000000000000000)'

Shapely can also integrate with other Python GIS packages using GeoJSON-like
dicts.

.. code-block:: pycon

    >>> import json
    >>> from shapely.geometry import mapping, shape
    >>> s = shape(json.loads('{"type": "Point", "coordinates": [0.0, 0.0]}'))
    >>> s
    <shapely.geometry.point.Point object at 0x...>
    >>> print(json.dumps(mapping(s)))
    {"type": "Point", "coordinates": [0.0, 0.0]}

Development and Testing
=======================

Dependencies for developing Shapely are listed in requirements-dev.txt. Cython
and Numpy are not required for production installations, only for development.
Use of a virtual environment is strongly recommended.

.. code-block:: console

    $ virtualenv .
    $ source bin/activate
    (env)$ pip install -r requirements-dev.txt
    (env)$ pip install -e .

The project uses pytest to run Shapely's suite of unittests and doctests.

.. code-block:: console

    (env)$ python -m pytest

Support
=======

Questions about using Shapely may be asked on the `GIS StackExchange
<https://gis.stackexchange.com/questions/tagged/shapely>`__ using the "shapely"
tag.

Bugs may be reported at https://github.com/shapely/shapely/issues.

Copyright & License
===================

Shapely is licensed under BSD 3-Clause license.
GEOS is available under the terms of GNU Lesser General Public License (LGPL) 2.1 at https://libgeos.org.
