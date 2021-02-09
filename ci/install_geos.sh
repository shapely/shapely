#!/bin/sh

# Build and install GEOS on a POSIX system, save cache for later
#
# This script requires environment variables to be set
#  - export GEOS_INSTALL=/path/to/cached/prefix -- to build or use as cache
#  - export GEOS_VERSION=3.7.3 or master -- to download and compile

set -e

if [ -z "$GEOS_INSTALL" ]; then
    echo "GEOS_INSTALL must be set"
    exit 1
elif [ -z "$GEOS_VERSION" ]; then
    echo "GEOS_VERSION must be set"
    exit 1
fi

# Create directories, if they don't exit
mkdir -p $GEOS_INSTALL

# Download and build GEOS outside other source tree
GEOS_BUILD=$HOME/geosbuild

prepare_geos_build_dir(){
  rm -rf $GEOS_BUILD
  mkdir -p $GEOS_BUILD
  cd $GEOS_BUILD
}

build_geos(){
    echo "Building geos-$GEOS_VERSION"
    mkdir build
    cd build
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$GEOS_INSTALL ..
    make -j 2
    ctest .
    make install
}

if [ "$GEOS_VERSION" = "master" ]; then
    prepare_geos_build_dir
    # use GitHub mirror
    git clone --depth 1 https://github.com/libgeos/geos.git geos-$GEOS_VERSION
    cd geos-$GEOS_VERSION
    git rev-parse HEAD > newrev.txt
    BUILD=no
    # Only build if nothing cached or if the GEOS revision changed
    if test ! -f $GEOS_INSTALL/rev.txt; then
        BUILD=yes
    elif ! diff newrev.txt $GEOS_INSTALL/rev.txt >/dev/null; then
        BUILD=yes
    fi
    if test "$BUILD" = "no"; then
        echo "Using cached install $GEOS_INSTALL"
    else
        cp newrev.txt $GEOS_INSTALL/rev.txt
        build_geos
    fi
else
    if [ -d "$GEOS_INSTALL/include/geos" ]; then
        echo "Using cached install $GEOS_INSTALL"
    else
        prepare_geos_build_dir
        wget -q http://download.osgeo.org/geos/geos-$GEOS_VERSION.tar.bz2
        tar xfj geos-$GEOS_VERSION.tar.bz2
        cd geos-$GEOS_VERSION
        build_geos
    fi
fi
