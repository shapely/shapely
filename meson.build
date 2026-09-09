# Entry point for shapely's meson-python (mesonpy) build backend

project(
  'shapely',
  'c', 'cython',
  version: run_command('python3', 'shapely/_version.py', check: true).stdout().strip(),
  license: 'BSD-3-Clause',
  meson_version: '>=1.2.1',  # keep in sync with [dependency-groups.build] in pyproject.toml
  default_options: [
    'buildtype=release',
    'c_std=c11',
    'warning_level=1',
  ],
)

###############################################################
# C setup
###############################################################

cc = meson.get_compiler('c')
_global_c_args = cc.get_supported_arguments(
  # TODO: resolve warnings
  '-Wno-cast-function-type',
  '-Wno-unused-parameter',
)
add_project_arguments(_global_c_args, language: 'c')

###############################################################
# Python setup
###############################################################

min_python_version = '3.11'  # keep in sync with [project.requires-python] in pyproject.toml

py = import('python').find_installation(pure: false)

python_version = py.language_version()
if python_version.version_compare(f'<@min_python_version@')
  error(f'Minimum supported Python version is @min_python_version@, found @python_version@')
endif

###############################################################
# GEOS setup
###############################################################

min_geos_version = '3.10'

# Previous shapely versions had multiple ways to specify GEOS dependency
# which are not supported (in principle). This warning can be removed someday.
ignored_env_vars = run_command(py, ['-c', '''
from os import environ
for name in ['GEOS_INCLUDE_PATH', 'GEOS_LIBRARY_PATH', 'GEOS_CONFIG']:
    if name in environ:
        print(name + "=" + environ[name])
'''], check: true).stdout().strip().split()
if ignored_env_vars != []
  warning(
    'These previously-supported environment variable(s) are not supported:',
    ignored_env_vars, '; use `cmake_prefix_path` ',
    'or `pkg_config_path` approach instead')
endif

# Enforce a dependency method search priority if built-in option is specified
geos_dep = dependency('', required: false)  # dummy to have dependency variable
if not geos_dep.found() and get_option('cmake_prefix_path').length() > 0
  # possibly configured with -Dcmake_prefix_path=/path/to/prefix
  geos_dep = dependency('geos', version: f'>=@min_geos_version@',
    method: 'cmake', modules: ['GEOS::geos_c'], required: false)
endif
if not geos_dep.found() and get_option('pkg_config_path').length() > 0
  # possibly configured with -Dpkg_config_path=/path/to/prefix/lib/pkgconfig
  geos_dep = dependency('geos', version: f'>=@min_geos_version@',
    method: 'pkg-config', required: false)
endif
if not geos_dep.found() # any method
  geos_dep = dependency('geos', version: f'>=@min_geos_version@',
    modules: ['GEOS::geos_c'])
endif

# avoid accidental use of non-reentrant functions
add_project_arguments('-DGEOS_USE_ONLY_R_API', language: 'c')

###############################################################
# Cython setup
###############################################################

cy = meson.get_compiler('cython')
if cy.version().version_compare('>=3.1.0')
  cython_args = ['-Xfreethreading_compatible=true']
endif

###############################################################
# NumPy setup
###############################################################

min_numpy_runtime_version = '1.26'  # keep in sync with [project.dependencies] in pyproject.toml
min_numpy_build_version = '1.26'  # keep in sync with [build-system.requires] in pyproject.toml

# get numpy dependency using different methods if numpy < 2
numpy_version = run_command(py,
  ['-c', 'import numpy as np; print(np.__version__)'], check: true).stdout().strip()
if numpy_version.version_compare('>=2.0')
  numpy_dep = dependency('numpy', version: f'>=@min_numpy_build_version@')
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
    include_directories: include_directories(numpy_incdir),
    version: numpy_version,
  )
endif

add_project_arguments('-DNPY_NO_DEPRECATED_API=0', language: 'c')

###############################################################
# Module sources
###############################################################

subdir('src')
subdir('shapely')
