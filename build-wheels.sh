#!/bin/bash

# Automation of this is a TODO. For now, it depends on manually built libraries
# as detailed in https://gist.github.com/sgillies/a8a2fb910a98a8566d0a.

export MACOSX_DEPLOYMENT_TARGET=10.6
export GEOS_CONFIG="/usr/local/bin/geos-config"

VERSION=$1

source $HOME/envs/pydotorg27/bin/activate
touch shapely/speedups/*.pyx
touch shapely/vectorized/*.pyx
CFLAGS="`$GEOS_CONFIG --cflags`" LDFLAGS="`$GEOS_CONFIG --libs`" python setup.py bdist_wheel -d wheels/$VERSION
source $HOME/envs/pydotorg34/bin/activate
touch shapely/speedups/*.pyx
touch shapely/vectorized/*.pyx
CFLAGS="`$GEOS_CONFIG --cflags`" LDFLAGS="`$GEOS_CONFIG --libs`" python setup.py bdist_wheel -d wheels/$VERSION

parallel delocate-wheel -w fixed_wheels/$VERSION --require-archs=intel -v {} ::: wheels/$VERSION/*.whl
parallel cp {} {.}.macosx_10_9_intel.macosx_10_9_x86_64.macosx_10_10_intel.macosx_10_10_x86_64.whl ::: fixed_wheels/$VERSION/*.whl
