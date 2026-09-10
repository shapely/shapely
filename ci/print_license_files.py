"""Print dynamic license-files (PEP 639) for build environment."""

from os import environ
from sys import platform

print("LICENSE.txt")
if bool(environ.get("CIBUILDWHEEL")):
    print("ci/wheelbuilder/LICENSE_GEOS")
    if platform.startswith("win"):
        print("ci/wheelbuilder/LICENSE_win32")
