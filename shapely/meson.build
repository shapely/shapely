py.extension_module(
  '_geometry_helpers',
  cython_gen.process('_geometry_helpers.pyx'),
  include_directories: ['../src', inc_np],
  dependencies : [py_dep, numpy_dep, geos_dep],
  install: true,
  subdir: 'shapely',
)

py.extension_module(
  '_geos',
  cython_gen.process('_geos.pyx'),
  include_directories: ['../src', inc_np],
  dependencies : [py_dep, numpy_dep, geos_dep],
  install: true,
  subdir: 'shapely',
)

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

subdir('algorithms')
subdir('geometry')
subdir('tests')
subdir('vectorized')
