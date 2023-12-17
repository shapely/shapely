
curl -L \
    https://github.com/geoarrow/geoarrow-c/archive/22794ce83fae1e2e99511508fa936c1e4cb115cb.zip \
    -o geoarrow.zip

unzip -d . geoarrow.zip

mkdir geoarrow-cmake
pushd geoarrow-cmake
cmake ../../geoarrow-c -DGEOARROW_BUNDLE=ON -DGEOARROW_USE_RYU=OFF -DGEOARROW_USE_FAST_FLOAT=OFF
cmake --build .
cmake --install . --prefix=../src
popd

rm geoarrow.zip
rm -rf geoarrow-c-*
rm -rf geoarrow-cmake
