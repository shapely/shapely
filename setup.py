
from setuptools import setup, Extension

# Get text from README.txt
readme_text = file('README.txt', 'rb').read()

setup(name          = 'Shapely',
      version       = '1.0a7',
      description   = 'Geospatial geometries, predicates, and operations',
      license       = 'LGPL',
      keywords      = 'geometry topology',
      author        = 'Sean Gillies',
      author_email  = 'sgillies@frii.com',
      maintainer    = 'Sean Gillies',
      maintainer_email  = 'sgillies@frii.com',
      url   = 'http://trac.gispython.org/projects/PCL/wiki/ShapeLy',
      long_description = readme_text,
      packages      = ['shapely', 'shapely.geometry'],
      install_requires = ['setuptools', 'ctypes'],
      tests_require = ['zope.testing', 'numpy'],
      classifiers   = [
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: GIS',
        ],
)

