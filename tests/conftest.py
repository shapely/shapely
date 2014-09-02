import sys

import pytest

def pytest_addoption(parser):
    parser.addoption("--with-speedups", action="store_true", default=False,
        help="Run tests with speedups.")

def pytest_runtest_setup(item):
    if item.config.getoption("--with-speedups"):
        import shapely.speedups
        if not shapely.speedups.available:
            print("Speedups have been demanded but are unavailable")
            sys.exit(1)
        shapely.speedups.enable()
        print("Speedups enabled for %s." % item.name)
