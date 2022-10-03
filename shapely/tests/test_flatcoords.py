import pytest

import shapely
from shapely.testing import assert_geometries_equal

from .common import all_types


@pytest.mark.parametrize("geom", all_types)
def test_roundtrip(geom):
    if geom.geom_type in ("GeometryCollection", "LinearRing"):
        with pytest.raises(ValueError):
            shapely.to_coordinates_offsets([geom, geom])
        return
    actual = shapely.from_coordinates_offsets(
        *shapely.to_coordinates_offsets([geom, geom])
    )
    assert_geometries_equal(actual, [geom, geom])
