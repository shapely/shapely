#!/bin/bash
# based on contribution by @rbuffat to Toblerity/Fiona
set -e

# Cache install directory from .travis.yml
CACHE_GEOS_INST=$HOME/geosinstall

# Create directories, if they don't exit
GEOS_INST_VERSION=$CACHE_GEOS_INST/geos-$GEOS_VERSION
mkdir -p $GEOS_INST_VERSION

function build_geos {
    echo "Building geos-$GEOS_VERSION"
    mkdir build
    cd build
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$GEOS_INST_VERSION ..
    make -j 2
    ctest .
    make install
}

if [ "$GEOS_VERSION" = "master" ]; then
    # use GitHub mirror
    git clone --depth 1 https://github.com/libgeos/geos.git geos-$GEOS_VERSION
    cd geos-$GEOS_VERSION
    git rev-parse HEAD > newrev.txt
    BUILD=no
    # Only build if nothing cached or if the GEOS revision changed
    if test ! -f $GEOS_INST_VERSION/rev.txt; then
        BUILD=yes
    elif ! diff newrev.txt $GEOS_INST_VERSION/rev.txt >/dev/null; then
        BUILD=yes
    fi
    if test "$BUILD" = "no"; then
        echo "Using cached install $GEOS_INST_VERSION"
    else
        cp newrev.txt $GEOS_INST_VERSION/rev.txt
        build_geos
    fi
else
    if [ -d "$GEOS_INST_VERSION/include/geos" ]; then
        echo "Using cached install $GEOS_INST_VERSION"
    else
        wget -q http://download.osgeo.org/geos/geos-$GEOS_VERSION.tar.bz2
        tar xfj geos-$GEOS_VERSION.tar.bz2
        cd geos-$GEOS_VERSION
        build_geos
    fi
fi
