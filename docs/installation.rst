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
during PyGEOS compilation is required on your system. Make sure that the dynamic linker paths are
set such that the libraries are found.

When using a binary wheel, conda, the Anaconda shell, or the OSGeo4W shell, this is automatically
the case and you can stop reading. In other cases this can be tricky, especially if you have
multiple GEOS installations next to each other. For PyGEOS this can be solved by prepending
the GEOS path before you start Python.

On Linux::

    $ export LD_LIBRARY_PATH=/path/to/geos/lib:$LD_LIBRARY_PATH
    $ sudo ldconfig  # refresh dynamic linker cache

On OSX::

    $ export DYLD_LIBRARY_PATH=/path/to/geos/lib:$DYLD_LIBRARY_PATH

On Windows::

    $ set PATH=C:\path\to\geos\bin;%PATH%

Note that setting environment variables like this is temporary. You will need to
repeat this every time before you want to use PyGEOS. Also, it will influence other
applications that are using GEOS; they may find a different GEOS version and crash.
