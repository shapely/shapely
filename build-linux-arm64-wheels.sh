#!/bin/bash
set -eu

# Checking for /.dockerenv is a hacky way to determine whether or not we're
# already running in a Docker container. Note that this is not guaranteed to
# exist in all versions and drivers and may need to be changed later.
if [ ! -e /.dockerenv ]; then
    docker build -f Dockerfile.arm64.wheels --pull -t shapely-wheelbuilder .
    exec docker run -v `pwd`:/io shapely-wheelbuilder "$@"
fi

ORIGINAL_PATH=$PATH
UNREPAIRED_WHEELS=/tmp/wheels

# Compile wheels
for PYBIN in /opt/python/cp3[789]*/bin; do
    PATH=${PYBIN}:$ORIGINAL_PATH
    python setup.py bdist_wheel -d ${UNREPAIRED_WHEELS}
done

# Bundle GEOS into the wheels
for whl in ${UNREPAIRED_WHEELS}/*.whl; do
    auditwheel repair ${whl} -w wheels
done
