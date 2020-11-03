Installation
============

Installation from PyPI
----------------------

PyGEOS is available as a binary distribution (wheel) for Linux, OSX and Windows platforms.
Install as follows::

    $ pip install pygeos


Installation using conda
------------------------

PyGEOS is available on the conda-forage channel. Install as follows::

    $ conda install pygeos --channel conda-forge


Installation with custom GEOS libary
------------------------------------

On Linux::

    $ sudo apt install libgeos-dev

On OSX::

    $ brew install geos

Make sure `geos-config` is available from you shell; append PATH if necessary::

    $ export PATH=$PATH:/path/to/dir/having/geos-config
    $ pip install pygeos --no-binary

We do not have a recipe for Windows platforms. The following steps should enable you
to build PyGEOS yourself:

- Get a C compiler applicable to your Python version (https://wiki.python.org/moin/WindowsCompilers)
- Download and install a GEOS binary (https://trac.osgeo.org/osgeo4w/)
- Set GEOS_INCLUDE_PATH and GEOS_LIBRARY_PATH environment variables
- Run ``pip install pygeos --no-binary``

Installation from source
------------------------

The same as above, but then instead of installing pygeos with pip, you clone the
package from Github::

    $ git clone git@github.com:pygeos/pygeos.git

Install Cython, which is required to build Cython extensions::

    $ pip install cython

Install it in development mode using `pip`::

    $ pip install -e .[test] --no-build-isolation

Run the unittests::

    $ pytest pygeos


Notes on GEOS discovery
-----------------------

If GEOS is installed, normally the ``geos-config`` command line utility
will be available, and ``pip install`` will find GEOS automatically.
But if needed, you can specify where PyGEOS should look for the GEOS library
before installing it:

On Linux / OSX::

    $ export GEOS_INCLUDE_PATH=$CONDA_PREFIX/Library/include
    $ export GEOS_LIBRARY_PATH=$CONDA_PREFIX/Library/lib

On Windows (assuming you are in a Visual C++ shell)::

    $ set GEOS_INCLUDE_PATH=%CONDA_PREFIX%\Library\include
    $ set GEOS_LIBRARY_PATH=%CONDA_PREFIX%\Library\lib
