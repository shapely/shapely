Installation
============

Built distributions
-------------------

Built distributions don't require compiling Shapely and its dependencies,
and can be installed using ``pip`` or ``conda``. In addition, Shapely is also
available via some system package management tools like apt or OSGeo4W.

Installation from PyPI
^^^^^^^^^^^^^^^^^^^^^^

Shapely is available as a binary distribution (wheel) for Linux, macOS, and
Windows platforms on `PyPI <https://pypi.org/project/shapely/>`__. The
distribution includes the most recent version of GEOS available at the time
of the Shapely release. Install the binary wheel with pip as follows::

    $ pip install shapely

Installation using conda
^^^^^^^^^^^^^^^^^^^^^^^^

Shapely is available on the conda-forge channel. Install as follows::

    $ conda install shapely --channel conda-forge

Installation of the development version using nightly wheels
------------------------------------------------------------

If you want to test the latest development version of Shapely, the easiest way
to get this version is by installing it from the Scientific Python index of
nightly wheel packages::

    python -m pip install --pre --upgrade --extra-index-url https://pypi.anaconda.org/scientific-python-nightly-wheels/simple shapely


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

If you've installed GEOS to a standard location on Linux or macOS, the
installation will automatically find it using CMake. If GEOS is installed
into a non-standard location, this will need to be specified. On Linux::

    $ GEOS_INSTALL=/path/to/geos-3.14.1
    $ pip install shapely --no-binary shapely -Csetup-args=-Dcmake_prefix_path="$GEOS_INSTALL"


Installation for local development
-----------------------------------

This is similar to installing with a custom GEOS binary, but then instead of
installing Shapely with pip from PyPI, you clone the package from GitHub:

.. code-block:: console

    $ git clone git@github.com:shapely/shapely.git
    $ cd shapely/

Local development with ``pip`` is enabled with an "editable" mode, where
edits to the source code become effective without the need of a new
installation step. As of Shapely 2.2, the build backend was changed to
meson-python, which requires `a few other instructions
<https://mesonbuild.com/meson-python/how-to-guides/editable-installs.html>`__,
such as ``--no-build-isolation``.


Virtual environments
^^^^^^^^^^^^^^^^^^^^

:external+python:doc:`Virtual environments <library/venv>` are recommended for
development. They can be created using Python or tools like
`uv <https://docs.astral.sh/uv/>`__.
For example, create and activate a virtual environment, then install the required
development packages:

.. code-block:: console

    $ python3 -m venv .venv
    $ source .venv/bin/activate
    $ pip install -r requirements-dev.txt

Install shapely in editable mode using ``pip``:

.. code-block:: console

    $ pip install --no-build-isolation -e .[test]

Conda environments
^^^^^^^^^^^^^^^^^^

Development within conda environments requires additional packages installed
with the ``conda`` command, including pre-built distributions of GEOS.
For example:

.. code-block:: console

    $ conda create -n shapely-dev --file requirements-dev.txt geos pip pytest

Activate the environment and install shapely in editable mode using ``pip``:
.. code-block:: console

    $ conda activate shapely-dev
    (shapely-dev) $ pip install --no-build-isolation -e .

Testing Shapely
---------------

Shapely can be tested using ``pytest``::

    $ pip install pytest  # or shapely[test]
    $ pytest --pyargs shapely.tests


GEOS discovery (compile time)
-----------------------------

GEOS is a core dependency built and installed with CMake. For example, build
and install it to ``$HOME/opt/geos`` as follows:

.. code-block:: console

    $ git clone git@github.com:libgeos/geos.git --depth 1
    $ cd geos
    $ cmake -GNinja -S . -B _build -DCMAKE_INSTALL_PREFIX=$HOME/opt/geos
    $ cmake --build _build
    $ cmake --install _build

If it is installed to a custom path, then it can be discovered by setting the
``cmake_prefix_path`` meson setup option:

.. code-block:: console

    $ pip install --no-build-isolation -e . -Csetup-args="-Dcmake_prefix_path=$HOME/opt/geos"

For Linux, if the GEOS library is not on the dynamic linker run-time path (i.e.
it was installed to a custom path), then the ``LD_LIBRARY_PATH`` environment
variable needs to be set before running, for example:

.. code-block:: console

    $ export LD_LIBRARY_PATH=$HOME/opt/geos/lib

Read more about GEOS runtime discovery in the next section.

.. note::

    Previous versions of shapely used several methods to discover GEOS,
    including ``geos-config``, ``GEOS_LIBRARY_PATH`` and ``GEOS_INCLUDE_PATH``.
    These methods are no longer supported since Shapely 2.2, to simplify
    the GEOS discovery process.

GEOS discovery (runtime)
------------------------

Shapely is dynamically linked to GEOS. This means that the same GEOS library
that was used during Shapely compilation is required on your system at runtime.
When using Shapely that was distributed as a binary wheel or through conda,
this is automatically the case and you can stop reading.

In other cases this can be tricky, especially if you have multiple GEOS
installations next to each other. We only include some guidelines here to
address this issue as this document is not intended as a general guide of
shared library discovery.

If you encounter exceptions like:

.. code-block:: none

   ImportError: libgeos_c.so.1: cannot open shared object file: No such file or directory

You will have to make the shared library file available to the Python
interpreter. There are in general four ways of making Python aware of the
location of shared library:

1. Copy the shared libraries into the ``shapely`` module directory (this is how
   Windows binary wheels work: they are distributed with the correct dlls in
   the ``shapely`` module directory)
2. Copy the shared libraries into the library directory of the Python
   interpreter (this is how Anaconda environments work)
3. Copy the shared libraries into some system location
   (``C:\Windows\System32``; ``/usr/local/lib``, this happens if you installed
   GEOS through ``apt`` or ``brew``)
4. Add the shared library location to a the dynamic linker path variable at
   runtime. (Advanced usage; Linux and macOS only; on Windows this method was
   deprecated in Python 3.8)

The filenames of the GEOS shared libraries are:

* On Linux: ``libgeos-*.so.*, libgeos_c-*.so.*``
* On macOS: ``libgeos.dylib, libgeos_c.dylib``
* On Windows: ``geos-*.dll, geos_c-*.dll``

Note that Shapely does not make use of any RUNPATH (RPATH) header. The location
of the GEOS shared library is not stored inside the compiled Shapely library.
