from setuptools import setup, Extension
from sys import version_info

# Require ctypes egg only for Python < 2.5
install_requires = ['setuptools']
if version_info[:2] < (2,5):
    install_requires.append('ctypes')

# Get text from README.txt
readme_text = file('README.txt', 'rb').read()

setup(name          = 'Shapely',
      version       = '1.0.12',
      description   = 'Geospatial geometries, predicates, and operations',
      license       = 'BSD',
      keywords      = 'geometry topology',
      author        = 'Sean Gillies',
      author_email  = 'sgillies@frii.com',
      maintainer    = 'Sean Gillies',
      maintainer_email  = 'sgillies@frii.com',
      url   = 'http://trac.gispython.org/lab/wiki/Shapely',
      long_description = readme_text,
      packages      = ['shapely', 'shapely.geometry'],
      data_files=[('DLLs', ['DLLs/geos.dll', 'DLLs/libgeos-3-0-0.dll']),],
      install_requires = install_requires,
      #tests_require = ['numpy'], -- not working with "tests" command
      test_suite = 'tests.test_suite',
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
