# Python and Cython source files for the "shapely.*" top-level modules

py.extension_module(
  '_geometry_helpers',
  '_geometry_helpers.pyx',
  cython_args: cython_args,
  include_directories: ['lib'],
  dependencies: [numpy_dep, geos_dep],
  install: true,
  subdir: 'shapely',
)

py.extension_module(
  '_geos',
  '_geos.pyx',
  cython_args: cython_args,
  include_directories: ['lib'],
  dependencies: [numpy_dep, geos_dep],
  install: true,
  subdir: 'shapely',
)

# top-level modules
py.install_sources([
    '__init__.py',
    'affinity.py',
    'conftest.py',
    'constructive.py',
    'coordinates.py',
    'coords.py',
    '_coverage.py',
    'creation.py',
    'decorators.py',
    '_enum.py',
    'errors.py',
    '_geometry.py',
    'geos.py',
    'io.py',
    'linear.py',
    'measurement.py',
    'ops.py',
    'plotting.py',
    'predicates.py',
    'prepared.py',
    '_ragged_array.py',
    'set_operations.py',
    'speedups.py',
    'strtree.py',
    'testing.py',
    'validation.py',
    '_version.py',
    'wkb.py',
    'wkt.py',
    '_geos.pxd',
    '_pygeos_api.pxd',
  ],
  subdir: 'shapely',
)

# Gather Python source files from these subdirectories
subdirs_list = [
  'algorithms',
  'geometry',
  'tests',
  'vectorized',
]
foreach subdir : subdirs_list
  install_subdir(subdir, install_dir: py.get_install_dir() / 'shapely')
endforeach
