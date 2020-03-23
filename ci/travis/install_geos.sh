#!/bin/bash
# based on contribution by @rbuffat to Toblerity/Fiona
set -e

# Cache install directory from .travis.yml
CACHEGEOSINST=$HOME/geosinstall

# Create directories, if they don't exit
GEOSINSTVERSION=$CACHEGEOSINST/geos-$GEOSVERSION
mkdir -p $GEOSINSTVERSION

function build_geos {
    echo "Building geos-$GEOSVERSION"
    mkdir build
    cd build
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$GEOSINSTVERSION ..
    make -j 2
    ctest .
    make install
}

if [ "$GEOSVERSION" = "master" ]; then
    # use GitHub mirror
    git clone --depth 1 https://github.com/libgeos/geos.git geos-$GEOSVERSION
    cd geos-$GEOSVERSION
    git rev-parse HEAD > newrev.txt
    BUILD=no
    # Only build if nothing cached or if the GEOS revision changed
    if test ! -f $GEOSINSTVERSION/rev.txt; then
        BUILD=yes
    elif ! diff newrev.txt $GEOSINSTVERSION/rev.txt >/dev/null; then
        BUILD=yes
    fi
    if test "$BUILD" = "no"; then
        echo "Using cached install $GEOSINSTVERSION"
    else
        cp newrev.txt $GEOSINSTVERSION/rev.txt
        build_geos
    fi
else
    if [ -d "$GEOSINSTVERSION/include/geos" ]; then
        echo "Using cached install $GEOSINSTVERSION"
    else
        wget -q http://download.osgeo.org/geos/geos-$GEOSVERSION.tar.bz2
        tar xfj geos-$GEOSVERSION.tar.bz2
        cd geos-$GEOSVERSION
        build_geos
    fi
fi
