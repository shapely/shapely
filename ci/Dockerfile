ARG BASEIMAGE=quay.io/pypa/manylinux_2_28_x86_64:latest
ARG GEOS_VERSION=3.13.1

FROM $BASEIMAGE
ARG GEOS_VERSION
ENV GEOS_VERSION=${GEOS_VERSION}
RUN curl -LOSs --retry 5 https://download.osgeo.org/geos/geos-$GEOS_VERSION.tar.bz2
RUN tar xfj geos-$GEOS_VERSION.tar.bz2 && rm -f geos-$GEOS_VERSION.tar.bz2
RUN cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -S geos-$GEOS_VERSION -B build
RUN cmake --build build -j 4
RUN cmake --install build
# Install again, where auditwheel can locate shared libs from LD_LIBRARY_PATH
RUN if [[ -n "$DEVTOOLSET_ROOTPATH" ]]; then cmake --install build --prefix $DEVTOOLSET_ROOTPATH; fi
RUN cd build && ctest --output-on-failure
RUN rm -rf build geos-$GEOS_VERSION
