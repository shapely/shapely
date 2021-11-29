Installation
============

Installation from PyPI
----------------------

PyGEOS is available as a binary distribution (wheel) for Linux, OSX and Windows platforms.
The distribution includes a GEOS version that was most recent at the time of the PyGEOS release.
Install the binary wheel with pip as follows::

    $ pip install pygeos


Installation using Anaconda
---------------------------

PyGEOS is available on the conda-forge channel. Install as follows::

    $ conda install pygeos --channel conda-forge


Installation with custom GEOS libary
------------------------------------

You may want to use a specific GEOS version or a GEOS distribution that is already present on
your system. In such cases you will need to compile PyGEOS yourself.

On Linux::

    $ sudo apt install libgeos-dev  # skip this if you already have GEOS
    $ pip install pygeos --no-binary

On OSX::

    $ brew install geos  # skip this if you already have GEOS
    $ pip install pygeos --no-binary

We do not have a recipe for Windows platforms. The following steps should enable you
to build PyGEOS yourself:

- Get a C compiler applicable to your Python version (https://wiki.python.org/moin/WindowsCompilers)
- Download and install a GEOS binary (https://trac.osgeo.org/osgeo4w/)
- Set GEOS_INCLUDE_PATH and GEOS_LIBRARY_PATH environment variables (see below for notes on GEOS discovery)
- Run ``pip install pygeos --no-binary``
- Make sure the GEOS .dll files are available on the PATH


Installation from source
------------------------

The same as installation with a custom GEOS binary, but then instead of installing pygeos
with pip, you clone the package from Github::

    $ git clone git@github.com:pygeos/pygeos.git

Install it in development mode using ``pip``::

    $ pip install -e .[test]


Testing PyGEOS
--------------

PyGEOS can be tested using ``pytest``::

    $ pip install pytest  # or pygeos[test]
    $ pytest --pyargs pygeos.tests


GEOS discovery (compile time)
-----------------------------

If GEOS is installed on Linux or OSX, normally the ``geos-config`` command line utility
will be available and ``pip`` will find GEOS automatically.
If the correct ``geos-config`` is not on the PATH, you can add it as follows (on Linux/OSX)::

    $ export PATH=/path/to/geos/bin:$PATH

Alternatively, you can specify where PyGEOS should look for GEOS (on Linux/OSX)::

    $ export GEOS_INCLUDE_PATH=/path/to/geos/include
    $ export GEOS_LIBRARY_PATH=/path/to/geos/lib

On Windows, there is no ``geos-config`` and the include and lib folders need to be
specified manually in any case::

    $ set GEOS_INCLUDE_PATH=C:\path\to\geos\include
    $ set GEOS_LIBRARY_PATH=C:\path\to\geos\lib

Common locations of GEOS (to be suffixed by ``lib``, ``include`` or ``bin``):

* Anaconda (Linux/OSX): ``$CONDA_PREFIX/Library``
* Anaconda (Windows): ``%CONDA_PREFIX%\Library``
* OSGeo4W (Windows): ``C:\OSGeo4W64``


GEOS discovery (runtime)
------------------------

PyGEOS is dynamically linked to GEOS. This means that the same GEOS library that was used
during PyGEOS compilation is required on your system at runtime. When using pygeos that was distributed
as a binary wheel or through conda, this is automatically the case and you can stop reading.

In other cases this can be tricky, especially if you have multiple GEOS installations next
to each other. We only include some guidelines here to address this issue as this document is
not intended as a general guide of shared library discovery.

If you encounter exceptions like:

.. code-block:: none

   ImportError: libgeos_c.so.1: cannot open shared object file: No such file or directory

You will have to make the shared library file available to the Python interpreter. There are in
general four ways of making Python aware of the location of shared library:

1. Copy the shared libraries into the pygeos module directory (this is how Windows binary wheels work:
   they are distributed with the correct dlls in the pygeos module directory)
2. Copy the shared libraries into the library directory of the Python interpreter (this is how
   Anaconda environments work)
3. Copy the shared libraries into some system location (``C:\Windows\System32``; ``/usr/local/lib``,
   this happens if you installed GEOS through ``apt`` or ``brew``)
4. Add the shared library location to a the dynamic linker path variable at runtime.
   (Advanced usage; Linux and OSX only; on Windows this method was deprecated in Python 3.8)

The filenames of the GEOS shared libraries are:

* On Linux: ``libgeos-*.so.*, libgeos_c-*.so.*``
* On OSX: ``libgeos.dylib, libgeos_c.dylib``
* On Windows: ``geos-*.dll, geos_c-*.dll``

Note that pygeos does not make use of any RUNPATH (RPATH) header. The location
of the GEOS shared library is not stored inside the compiled PyGEOS library.
