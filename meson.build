project(
  'shapely',
  'c', 'cython',
  version: '2.2.0.dev0',
  license: 'BSD-3',
  meson_version: '>=1.2.1',
  default_options: [
    'buildtype=release',
    'c_std=c11',
    'warning_level=2',
  ],
)

py = import('python').find_installation(pure: false)
py_dep = py.dependency()

min_geos_version = '3.10'
min_numpy_version = '1.23'  # keep in sync with pyproject.toml and NPY_TARGET_VERSION
min_python_version = '3.11'  # keep in sync with pyproject.toml

python_version = py.language_version()
if python_version.version_compare(f'<@min_python_version@')
  error(f'Minimum supported Python version is @min_python_version@, found @python_version@')
endif

# silence warnings
add_project_arguments('-DNPY_NO_DEPRECATED_API=0', language : 'c')

# see also min_numpy_version
add_project_arguments('-DNPY_TARGET_VERSION=NPY_1_23_API_VERSION', language : 'c')

# Previous shapely versions had multiple ways to specify GEOS dependency
# which are not supported (in principle)
ignored_env_vars = run_command(py, ['-c', '''
from os import environ
for name in ['GEOS_INCLUDE_PATH', 'GEOS_LIBRARY_PATH', 'GEOS_CONFIG']:
    if name in environ:
        print(name + "=" + environ[name])
'''],
  check: true,
).stdout().strip().split()
if ignored_env_vars != []
  warning(
    'These previously-supported environment variable(s) are not supported:',
    ignored_env_vars,
    '; use `cmake_prefix_path` approach instead')
endif

# A custom GEOS installation can be provided with configuration option:
# -Dcmake_prefix_path=/path/to/geos-install

geos_dep = dependency('geos', version : f'>=@min_geos_version@',
  method : 'cmake',  modules : ['GEOS::geos_c'])

# avoid accidental use of non-reentrant functions
add_project_arguments('-DGEOS_USE_ONLY_R_API', language : 'c')

cy = meson.get_compiler('cython')

# generator() doesn't accept compilers, only found programs - cast it.
cython = find_program(cy.cmd_array()[0])
cython_args = ['-3', '--fast-fail', '--output-file', '@OUTPUT@', '--include-dir', '@BUILD_ROOT@', '@INPUT@']
if cy.version().version_compare('>=3.1.0')
  cython_args += ['-Xfreethreading_compatible=True']
endif
cython_gen = generator(cython, arguments : cython_args, output : '@BASENAME@.c')

# get numpy dependency using different methods if numpy < 2
numpy_version = run_command(py,
  ['-c', 'import numpy as np; print(np.__version__)'], check: true).stdout().strip()
if numpy_version.version_compare('>=2.0')
  numpy_dep = dependency('numpy', version : f'>=@min_numpy_version@')
else  # support numpy < 2
  numpy_incdir = run_command(py, ['-c', '''
import os
import numpy as np
try:
    # Check if include directory is inside the shapely dir e.g. a venv created
    # inside the shapely dir. If so, convert it to a relative path.
    incdir = os.path.relpath(np.get_include())
except Exception:
    incdir = np.get_include()
print(incdir)
'''], check: true).stdout().strip()
  numpy_dep = declare_dependency(
    include_directories : include_directories(numpy_incdir),
    version : numpy_version,
  )
endif

subdir('src')
subdir('shapely')
