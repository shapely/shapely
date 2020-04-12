import pytest

from shapely.geos import geos_version
from shapely.wkt import loads as load_wkt
from shapely.geometry import Polygon

from shapely.validation import make_valid

requires_geos_38 = pytest.mark.skipif(geos_version < (3, 8, 0), reason='GEOS >= 3.8.0 is required.')


@requires_geos_38
def test_make_valid_invalid_input():
    geom = Polygon([(0, 0), (0, 2), (1, 1), (2, 2), (2, 0), (1, 1), (0, 0)])
    valid = make_valid(geom)
    assert len(valid) == 2
    assert all(geom.type == 'Polygon' for geom in valid)


@requires_geos_38
def test_make_valid_input():
    geom = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    valid = make_valid(geom)
    assert id(valid) == id(geom)

