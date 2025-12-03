project(
  'shapely',
  'c', 'cython',
  version: '2.2.0.dev0',
  license: 'BSD-3',
  meson_version: '>=1.2.1',
  default_options: ['buildtype=release', 'c_std=c11', 'warning_level=2'],
)

py = import('python').find_installation(pure: false)
py_dep = py.dependency()
cy = meson.get_compiler('cython')

# generator() doesn't accept compilers, only found programs - cast it.
cython = find_program(cy.cmd_array()[0])
cython_args = ['-3', '--fast-fail', '--output-file', '@OUTPUT@', '--include-dir', '@BUILD_ROOT@', '@INPUT@']
cython_gen = generator(cython,
  arguments : cython_args,
  output : '@BASENAME@.c',
)

numpy_dep = dependency('numpy')
incdir_numpy = run_command(
    py,
    [
        '-c',
        '''
import os
import numpy as np
try:
    # Check if include directory is inside the shapely dir
    # e.g. a venv created inside the shapely dir
    # If so, convert it to a relative path
    incdir = os.path.relpath(np.get_include())
except Exception:
    incdir = np.get_include()
print(incdir)
     ''',
    ],
    check: true,
).stdout().strip()

inc_np = include_directories(incdir_numpy)

geos_dep = dependency('geos', required: false, version : '>=3.10')

add_project_arguments('-DNPY_NO_DEPRECATED_API=0', language : 'c')
add_project_arguments('-DNPY_TARGET_VERSION=NPY_1_23_API_VERSION', language : 'c')

subdir('src')
subdir('shapely')
