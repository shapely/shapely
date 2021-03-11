import pytest

import pytest

from shapely.errors import ShapelyDeprecationWarning
from shapely.geometry import Point, Polygon
from shapely.geos import geos_version
from shapely.strtree import STRtree


@pytest.mark.skipif(geos_version < (3, 6, 0), reason="GEOS 3.6.0 required")
@pytest.mark.parametrize(
    "geoms",
    [
        [
            Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
            Polygon([(0, 2), (1, 2), (1, 3), (0, 3)]),
            Point(0, 0.5),
        ]
    ],
)
@pytest.mark.parametrize("query_geom", [Point(0, 0.4)])
def test_nearest_geom(geoms, query_geom):
    with pytest.warns(ShapelyDeprecationWarning):
        tree = STRtree(geoms)
    result = tree.nearest(query_geom)
    assert result.geom_type == "Point"
    assert result.x == 0.0
    assert result.y == 0.5


@pytest.mark.skipif(geos_version < (3, 6, 0), reason="GEOS 3.6.0 required")
@pytest.mark.parametrize(
    "geoms",
    [
        [
            Point(0, 0.5),
            Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
            Polygon([(0, 2), (1, 2), (1, 3), (0, 3)]),
        ]
    ],
)
@pytest.mark.parametrize("values", [["Ahoy!", "Hi!", "Hi!"]])
@pytest.mark.parametrize("query_geom", [Point(0, 0.4)])
def test_nearest_value(geoms, values, query_geom):
    with pytest.warns(ShapelyDeprecationWarning):
        tree = STRtree(zip(geoms, values))
    tree.nearest(query_geom) == "Ahoy!"