:: Build and install GEOS on Windows system, save cache for later
::
:: This script requires environment variables to be set
::  - set GEOS_INSTALL=C:\path\to\cached\prefix -- to build or use as cache
::  - set GEOS_VERSION=3.7.3 -- to download and compile

if exist %GEOS_INSTALL% (
  echo Using cached %GEOS_INSTALL%
) else (
  echo Building %GEOS_INSTALL%

  curl -fsSO http://download.osgeo.org/geos/geos-%GEOS_VERSION%.tar.bz2
  7z x geos-%GEOS_VERSION%.tar.bz2
  7z x geos-%GEOS_VERSION%.tar
  cd geos-%GEOS_VERSION% || exit /B 1

  pip install ninja
  cmake --version

  mkdir build
  cd build
  cmake -GNinja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=%GEOS_INSTALL% .. || exit /B 2
  cmake --build . || exit /B 3
  :: ctest . || exit /B 4
  ctest .
  cmake --install . || exit /B 5
  cd ..
)
