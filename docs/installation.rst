Installation
============

Built distributions
-------------------

Built distributions don't require compiling Shapely and its dependencies,
and can be installed using ``pip`` or ``conda``. In addition, Shapely is also
available via some system package management tools like apt.

Installation from PyPI
^^^^^^^^^^^^^^^^^^^^^^

Shapely is available as a binary distribution (wheel) for Linux, macOS, and
Windows platforms on `PyPI <https://pypi.org/project/Shapely/>`__. The
distribution includes the most recent version of GEOS available at the time
of the Shapely release. Install the binary wheel with pip as follows::

    $ pip install shapely

Installation using conda
^^^^^^^^^^^^^^^^^^^^^^^^

Shapely is available on the conda-forge channel. Install as follows::

    $ conda install shapely --channel conda-forge


Installation from source with custom GEOS library
-------------------------------------------------

You may want to use a specific GEOS version or a GEOS distribution that is
already present on your system (for compatibility with other modules that
depend on GEOS, such as cartopy or osgeo.ogr). In such cases you will need to
ensure the GEOS library is installed on your system and then compile Shapely
from source yourself, by directing pip to ignore the binary wheels.

On Linux::

    $ sudo apt install libgeos-dev  # skip this if you already have GEOS
    $ pip install shapely --no-binary shapely

On macOS::

    $ brew install geos  # skip this if you already have GEOS
    $ pip install shapely --no-binary shapely

If you've installed GEOS to a standard location on Linux or macOS, the installation will automatically
find it using ``geos-config``. See the notes below on GEOS discovery at compile time
to configure this.

We do not have a recipe for Windows platforms. The following steps should enable you
to build Shapely yourself:

- Get a C compiler applicable to your Python version (https://wiki.python.org/moin/WindowsCompilers)
- Download and install a GEOS binary (https://trac.osgeo.org/osgeo4w/)
- Set GEOS_INCLUDE_PATH and GEOS_LIBRARY_PATH environment variables (see below for notes on GEOS discovery)
- Run ``pip install shapely --no-binary``
- Make sure the GEOS .dll files are available on the PATH


Installation for local development
-----------------------------------

This is similar to installing with a custom GEOS binary, but then instead of installing
Shapely with pip from PyPI, you clone the package from Github::

    $ git clone git@github.com:shapely/shapely.git
    $ cd shapely/

Install it in development mode using ``pip``::

    $ pip install -e .[test]

For development, use of a virtual environment is strongly recommended. For example
using ``venv``:

.. code-block:: console

    $ python3 -m venv .
    $ source bin/activate
    (env) $ pip install -e .[test]

Or using ``conda``:

.. code-block:: console

    $ conda create -n env python=3 geos numpy cython pytest
    $ conda activate env
    (env) $ pip install -e .

Testing Shapely
---------------

Shapely can be tested using ``pytest``::

    $ pip install pytest  # or shapely[test]
    $ pytest --pyargs shapely.tests


GEOS discovery (compile time)
-----------------------------

If GEOS is installed on Linux or macOS, the ``geos-config`` command line utility
should be available and ``pip`` will find GEOS automatically.
If the correct ``geos-config`` is not on the PATH, you can add it as follows (on Linux/macOS)::

    $ export PATH=/path/to/geos/bin:$PATH

Alternatively, you can specify where Shapely should look for GEOS library and
header files using environment variables (on Linux/macOS)::

    $ export GEOS_INCLUDE_PATH=/path/to/geos/include
    $ export GEOS_LIBRARY_PATH=/path/to/geos/lib

On Windows, there is no ``geos-config`` and the include and lib folders need to be
specified manually in any case::

    $ set GEOS_INCLUDE_PATH=C:\path\to\geos\include
    $ set GEOS_LIBRARY_PATH=C:\path\to\geos\lib

Common locations of GEOS (to be suffixed by ``lib``, ``include`` or ``bin``):

* Anaconda (Linux/macOS): ``$CONDA_PREFIX/Library``
* Anaconda (Windows): ``%CONDA_PREFIX%\Library``
* OSGeo4W (Windows): ``C:\OSGeo4W64``


GEOS discovery (runtime)
------------------------

Shapely is dynamically linked to GEOS. This means that the same GEOS library that was used
during Shapely compilation is required on your system at runtime. When using Shapely that was distributed
as a binary wheel or through conda, this is automatically the case and you can stop reading.

In other cases this can be tricky, especially if you have multiple GEOS installations next
to each other. We only include some guidelines here to address this issue as this document is
not intended as a general guide of shared library discovery.

If you encounter exceptions like:

.. code-block:: none

   ImportError: libgeos_c.so.1: cannot open shared object file: No such file or directory

You will have to make the shared library file available to the Python interpreter. There are in
general four ways of making Python aware of the location of shared library:

1. Copy the shared libraries into the ``shapely`` module directory (this is how Windows binary wheels work:
   they are distributed with the correct dlls in the ``shapely`` module directory)
2. Copy the shared libraries into the library directory of the Python interpreter (this is how
   Anaconda environments work)
3. Copy the shared libraries into some system location (``C:\Windows\System32``; ``/usr/local/lib``,
   this happens if you installed GEOS through ``apt`` or ``brew``)
4. Add the shared library location to a the dynamic linker path variable at runtime.
   (Advanced usage; Linux and macOS only; on Windows this method was deprecated in Python 3.8)

The filenames of the GEOS shared libraries are:

* On Linux: ``libgeos-*.so.*, libgeos_c-*.so.*``
* On macOS: ``libgeos.dylib, libgeos_c.dylib``
* On Windows: ``geos-*.dll, geos_c-*.dll``

Note that Shapely does not make use of any RUNPATH (RPATH) header. The location
of the GEOS shared library is not stored inside the compiled Shapely library.
