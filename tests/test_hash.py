import pytest

from shapely.geometry import GeometryCollection, LineString, MultiPoint, Point


@pytest.mark.parametrize(
    "geom",
    [
        Point(1, 2),
        MultiPoint([(1, 2), (3, 4)]),
        LineString([(1, 2), (3, 4)]),
        Point(0, 0).buffer(1.0),
        GeometryCollection([Point(1, 2), LineString([(1, 2), (3, 4)])]),
    ],
    ids=[
        "Point",
        "MultiPoint",
        "LineString",
        "Polygon",
        "GeometryCollection",
    ],
)
def test_hash(geom):
    with pytest.raises(TypeError, match="unhashable type"):
        hash(geom)
