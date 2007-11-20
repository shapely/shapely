
from setuptools import setup, Extension

# Get text from README.txt
readme_text = file('README.txt', 'rb').read()

setup(name          = 'Shapely',
      version       = '1.0b1',
      description   = 'Geospatial geometries, predicates, and operations',
      license       = 'BSD',
      keywords      = 'geometry topology',
      author        = 'Sean Gillies',
      author_email  = 'sgillies@frii.com',
      maintainer    = 'Sean Gillies',
      maintainer_email  = 'sgillies@frii.com',
      url   = 'http://trac.gispython.org/projects/PCL/wiki/Shapely',
      long_description = readme_text,
      packages      = ['shapely'],
      install_requires = ['setuptools', 'ctypes'],
      #tests_require = ['numpy'],
      test_suite = 'tests.test_suite',
      classifiers   = [
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: GIS',
        ],
)

