
rm src/geoarrow*

GEOARROW_C_REF="d3ad6e4fe28fc88d714b3805cc425462125487f4"
GEOARROW_C_GEOS_REF="125d865179af4002ab1932822e26e292909abb98"

curl -L \
    "https://github.com/geoarrow/geoarrow-c/archive/${GEOARROW_C_REF}.zip" \
    -o geoarrow.zip

unzip -d . geoarrow.zip

CMAKE_DIR="$(dirname geoarrow-c-*/**/CMakeLists.txt)"

mkdir geoarrow-cmake
pushd geoarrow-cmake
cmake ${CMAKE_DIR} -DGEOARROW_BUNDLE=ON -DGEOARROW_USE_RYU=OFF -DGEOARROW_USE_FAST_FLOAT=OFF
cmake --build .
cmake --install . --prefix=../src
popd

rm geoarrow.zip
rm -rf geoarrow-c-*
rm -rf geoarrow-cmake

for f in geoarrow_geos.h geoarrow_geos.c; do
  curl -L \
    "https://raw.githubusercontent.com/geoarrow/geoarrow-c-geos/${GEOARROW_C_GEOS_REF}/src/geoarrow_geos/${f}" \
    -o "src/${f}"
done
