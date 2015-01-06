#!/bin/bash

# Dependent on the Kyngchaos Frameworks:
# http://www.kyngchaos.com/software/frameworks

export GEOS_CONFIG="/Library/Frameworks/GEOS.framework/Versions/3/unix/bin/geos-config"
CFLAGS="`$GEOS_CONFIG --cflags`" LDFLAGS="`$GEOS_CONFIG --clibs`" python setup.py bdist_wheel
delocate-wheel -w fixed_wheels --require-archs=intel -v dist/Shapely-1.5.2-cp27-none-macosx_10_6_intel.whl
