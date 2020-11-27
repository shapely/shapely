#!/usr/bin/env python

# Build or install Shapely distributions
#
# This script has two different uses.
#
# 1) Installing from a source distribution, whether via
#
#      ``python setup.py install``
#
#    after downloading a source distribution, or
#
#      ``pip install shapely``
#
#    on a platform for which pip cannot find a wheel. This will most
#    often be the case for Linux, since the project is not yet
#    publishing Linux wheels. This will never be the case on Windows and
#    rarely the case on OS X; both are wheels-first platforms.
#
# 2) Building distributions (source or wheel) from a repository. This
#    includes using Cython to generate C source for the speedups and
#    vectorize modules from Shapely's .pyx files.
#
# On import, Shapely loads a GEOS shared library. GEOS is a run time
# requirement. Additionally, the speedups and vectorized C extension
# modules need GEOS headers and libraries to be built. Shapely versions
# >=1.3 require GEOS >= 3.3.
#
# For the first use case (see 1, above), we aim to treat GEOS as if it
# were a Python requirement listed in ``install_requires``. That is, in
# an environment with Shapely 1.2.x and GEOS 3.2, the command ``pip
# install shapely >=1.3 --no-use-wheel`` (whether wheels are explicitly
# opted against or are not published for the platform) should fail with
# a warning and advice to upgrade GEOS to >=3.3.
#
# In case 1, the environment's GEOS version is determined by executing
# the geos-config script. If the GEOS version returned by that script is
# incompatible with the Shapely source distribution or no geos-config
# script can be found, this setup script will fail.
#
# For the second use case (see 2, distribution building, above), we
# allow the requirements to be loosened. If this script finds that the
# environment variable NO_GEOS_CHECK is set, geos-config will not be
# executed and no attempt will be made to enforce requirements as in the
# second case.
#
# For both cases, a geos-config not in the environment's $PATH may be
# used by setting the environment variable GEOS_CONFIG to the path to
# a geos-config script.
#
# NB: within this setup scripts, software versions are evaluated according
# to https://www.python.org/dev/peps/pep-0440/.

import errno
import glob
import itertools as it
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
from setuptools import setup
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext as distutils_build_ext
from distutils.errors import CCompilerError, DistutilsExecError, \
    DistutilsPlatformError

from _vendor.packaging.version import Version

# Get geos_version from GEOS dynamic library, which depends on
# GEOS_LIBRARY_PATH and/or GEOS_CONFIG environment variables
from shapely._buildcfg import geos_version_string, geos_version, \
        geos_config, get_geos_config

logging.basicConfig()
log = logging.getLogger(__file__)

# python -W all setup.py ...
if 'all' in sys.warnoptions:
    log.level = logging.DEBUG


class GEOSConfig(object):
    """Interface to config options from the `geos-config` utility
    """

    def __init__(self, cmd):
        self.cmd = cmd

    def get(self, option):
        try:
            stdout, stderr = subprocess.Popen(
                [self.cmd, option],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
        except OSError as ex:
            # e.g., [Errno 2] No such file or directory
            raise OSError("Could not find geos-config script")
        if stderr and not stdout:
            raise ValueError(stderr.strip())
        result = stdout.decode('ascii').strip()
        log.debug('%s %s: %r', self.cmd, option, result)
        return result

    def version(self):
        match = re.match(r'(\d+)\.(\d+)\.(\d+)', self.get('--version').strip())
        return tuple(map(int, match.groups()))

# Get the version from the shapely module.
shapely_version = None
with open('shapely/__init__.py', 'r') as fp:
    for line in fp:
        if line.startswith("__version__"):
            shapely_version = Version(
                line.split("=")[1].strip().strip("\"'"))
            break

if not shapely_version:
    raise ValueError("Could not determine Shapely's version")

# Allow GEOS_CONFIG to be bypassed in favor of CFLAGS and LDFLAGS
# vars set by build environment.
if os.environ.get('NO_GEOS_CONFIG'):
    geos_config = None
else:
    geos_config = GEOSConfig(os.environ.get('GEOS_CONFIG', 'geos-config'))

# Fail installation if the GEOS shared library does not meet the minimum
# version. We ship it with Shapely for Windows, so no need to check on
# that platform.
geos_version = None
if geos_config and not os.environ.get('NO_GEOS_CHECK') or sys.platform == 'win32':
    try:
        log.info(
            "Shapely >= 1.3 requires GEOS >= 3.3. "
            "Checking for GEOS version...")
        geos_version = geos_config.version()
        log.info("Found GEOS version: %s", geos_version)
        if (set(sys.argv).intersection(['install', 'build', 'build_ext']) and
                shapely_version >= Version("1.3") and geos_version < (3, 3)):
            log.critical(
                "Shapely >= 1.3 requires GEOS >= 3.3. "
                "Install GEOS 3.3+ and reinstall Shapely.")
            sys.exit(1)
    except OSError as exc:
        log.warning(
            "Failed to determine system's GEOS version: %s. "
            "Installation continuing. GEOS version will be "
            "checked on import of shapely.", exc)

with open('VERSION.txt', 'w') as fp:
    fp.write(str(shapely_version))

with open('README.rst', 'r') as fp:
    readme = fp.read()

with open('CREDITS.txt', 'r', encoding='utf-8') as fp:
    credits = fp.read()

with open('CHANGES.txt', 'r') as fp:
    changes = fp.read()

long_description = readme + '\n\n' + credits + '\n\n' + changes

extra_reqs = {
    'test': ['pytest', 'pytest-cov'],
}
extra_reqs['all'] = list(it.chain.from_iterable(extra_reqs.values()))

# Make a dict of setup arguments. Some items will be updated as
# the script progresses.
setup_args = dict(
    name                = 'Shapely',
    version             = str(shapely_version),
    description         = 'Geometric objects, predicates, and operations',
    license             = 'BSD',
    keywords            = 'geometry topology gis',
    author              = 'Sean Gillies',
    author_email        = 'sean.gillies@gmail.com',
    maintainer          = 'Sean Gillies',
    maintainer_email    = 'sean.gillies@gmail.com',
    url                 = 'https://github.com/Toblerity/Shapely',
    long_description    = long_description,
    packages            = [
        'shapely',
        'shapely.geometry',
        'shapely.algorithms',
        'shapely.examples',
        'shapely.vectorized',
    ],
    classifiers         = [
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: GIS',
    ],
    cmdclass           = {},
    python_requires    = '>=3.5',
    extras_require     = extra_reqs,
    package_data={
        'shapely': ['shapely/_geos.pxi']},
    include_package_data=True
)

# Add DLLs for Windows.
if sys.platform == 'win32':
    try:
        os.mkdir('shapely/DLLs')
    except OSError as ex:
        if ex.errno != errno.EEXIST:
            raise
    if '(AMD64)' in sys.version:
        for dll in glob.glob('DLLs_AMD64_VC9/*.dll'):
            shutil.copy(dll, 'shapely/DLLs')
    else:
        for dll in glob.glob('DLLs_x86_VC9/*.dll'):
            shutil.copy(dll, 'shapely/DLLs')
    setup_args['package_data']['shapely'].append('shapely/DLLs/*.dll')


setup(**setup_args)
