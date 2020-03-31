REM This script is called from appveyor.yml

if exist %GEOSINSTALL% (
  echo Using cached %GEOSINSTALL%
) else (
  echo Building %GEOSINSTALL%

  cd C:\projects

  curl -fsSO http://download.osgeo.org/geos/geos-%GEOSVERSION%.tar.bz2
  7z x geos-%GEOSVERSION%.tar.bz2
  7z x geos-%GEOSVERSION%.tar
  cd geos-%GEOSVERSION% || exit /B 5

  pip install ninja
  cmake --version

  mkdir build
  cd build
  cmake -GNinja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=%GEOSINSTALL% .. || exit /B 1
  cmake --build . --config Release || exit /B 2
  ctest . --config Release || exit /B 3
  cmake --install . --config Release || exit /B 4
  cd ..
)
