=======
Shapely
=======

|github-actions| |coveralls|

.. |github-actions| image:: https://github.com/shapely/shapely/workflows/Tests/badge.svg?branch=main
   :target: https://github.com/shapely/shapely/actions?query=branch%3Amain

.. |coveralls| image:: https://coveralls.io/repos/github/shapely/shapely/badge.svg?branch=main
   :target: https://coveralls.io/github/shapely/shapely?branch=main

Manipulation and analysis of geometric objects in the Cartesian plane.

.. image:: https://c2.staticflickr.com/6/5560/31301790086_b3472ea4e9_c.jpg
   :width: 800
   :height: 378

Shapely is a BSD-licensed Python package for manipulation and analysis of
planar geometric objects. It is based on the widely deployed `GEOS
<https://libgeos.org/>`__ (the engine of `PostGIS
<https://postgis.net/>`__) and `JTS
<https://locationtech.github.io/jts/>`__ (from which GEOS is ported)
libraries. Shapely is not concerned with data formats or coordinate systems,
but can be readily integrated with packages that are. For more details, see:

* `Shapely GitHub repository <https://github.com/shapely/shapely>`__
* `Shapely documentation and manual <https://shapely.readthedocs.io/en/latest/>`__

Usage
=====

Here is the canonical example of building an approximately circular patch by
buffering a point.

.. code-block:: pycon

    >>> from shapely.geometry import Point
    >>> patch = Point(0.0, 0.0).buffer(10.0)
    >>> patch
    <shapely.geometry.polygon.Polygon object at 0x...>
    >>> patch.area
    313.65484905459385

See the manual for more examples and guidance.

Requirements
============

Shapely 2.0 requires

* Python >=3.6
* GEOS >=3.5
* NumPy >=1.13

Installing Shapely
==================

It is recommended to install shapely using one of the available built
distributions, for example using ``pip`` or ``conda``:

.. code-block:: console

    $ pip install shapely
    # or using conda
    $ conda install shapely --channel conda-forge

See the `installation documentation <https://shapely.readthedocs.io/en/latest/installation.html>`__
for more details (installing from source, using a custom GEOS install, development version, etc).

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

Support
=======

Questions about using Shapely may be asked on the `GIS StackExchange
<https://gis.stackexchange.com/questions/tagged/shapely>`__ using the "shapely"
tag.

Bugs may be reported at https://github.com/shapely/shapely/issues.
