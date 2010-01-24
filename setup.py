import warnings

try:
    from distribute_setup import use_setuptools
    use_setuptools()
except:
    warnings.warn("Failed to import distribute_setup", ImportWarning)

from setuptools import setup, Extension
import sys

# Require ctypes egg only for Python < 2.5
install_requires = []
if sys.version_info[:2] < (2,5):
    install_requires.append('ctypes')

# Get text from README.txt
readme_text = file('README.txt', 'rb').read()

setup_args = dict(
    name          = 'Shapely',
    version       = '1.2a4',
    description   = 'Geospatial geometries, predicates, and operations',
    license       = 'BSD',
    keywords      = 'geometry topology',
    author        = 'Sean Gillies',
    author_email  = 'sean.gillies@gmail.com',
    maintainer    = 'Sean Gillies',
    maintainer_email  = 'sean.gillies@gmail.com',
    url   = 'http://trac.gispython.org/lab/wiki/Shapely',
    long_description = readme_text,
    packages      = ['shapely', 'shapely.geometry'],
    install_requires = install_requires,
    test_suite = 'shapely.tests.test_suite',
    classifiers   = [
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: GIS',
        ],
    )

# Add DLLs for Windows
if sys.platform == 'win32':
    setup_args.update(
        data_files=[('DLLs', ['DLLs/geos.dll', 'DLLs/libgeos-3-0-0.dll']),]
        )

setup(**setup_args)
