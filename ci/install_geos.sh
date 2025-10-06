#!/bin/bash

# Build and install GEOS on a POSIX system, save cache for later
#
# This script requires environment variables to be set
#  - export GEOS_INSTALL=/path/to/cached/prefix -- to build or use as cache
#  - export GEOS_VERSION=3.7.3 or main -- to download and compile
pushd .

set -e

if [ -z "$GEOS_INSTALL" ]; then
    echo "GEOS_INSTALL must be set"
    exit 1
elif [ -z "$GEOS_VERSION" ]; then
    echo "GEOS_VERSION must be set"
    exit 1
fi

# Create directories, if they don't exist
mkdir -p $GEOS_INSTALL

# Download and build GEOS outside other source tree
if [ -z "$GEOS_BUILD" ]; then
    GEOS_BUILD=$HOME/geosbuild
fi

prepare_geos_build_dir(){
  rm -rf $GEOS_BUILD
  mkdir -p $GEOS_BUILD
  cd $GEOS_BUILD
}

build_geos(){
    echo "Installing cmake"
    pip install cmake

    echo "Building geos-$GEOS_VERSION"
    rm -rf build
    mkdir build
    cd build
    # Use Ninja on Windows, otherwise, use the platform's default
    if [ "$RUNNER_OS" = "Windows" ]; then
        export CMAKE_GENERATOR=Ninja
    fi
    # Avoid building tests, depends on version
    case ${GEOS_VERSION} in
        3.7.*)
            BUILD_TESTING="";;
        3.8.*)
            BUILD_TESTING="-DBUILD_TESTING=ON";;
        *)
            BUILD_TESTING="-DBUILD_TESTING=OFF";;
    esac
    cmake \
        -D CMAKE_BUILD_TYPE=Release \
        -D CMAKE_INSTALL_PREFIX=${GEOS_INSTALL} \
        -D CMAKE_INSTALL_LIBDIR=lib \
        -D CMAKE_INSTALL_NAME_DIR=${GEOS_INSTALL}/lib \
        ${BUILD_TESTING} \
        ..
    cmake --build . -j 4
    cmake --install .
}

if [ -d "$GEOS_INSTALL/include/geos" ]; then
    echo "Using cached install $GEOS_INSTALL"
else
    if [ "$GEOS_VERSION" = "main" ]; then
        # Expect the CI to have put the latest checkout in GEOS_BUILD
        cd $GEOS_BUILD
        build_geos
    else
        prepare_geos_build_dir
        curl -OL http://download.osgeo.org/geos/geos-$GEOS_VERSION.tar.bz2
        tar xfj geos-$GEOS_VERSION.tar.bz2
        cd geos-$GEOS_VERSION
        build_geos
    fi
fi

popd
