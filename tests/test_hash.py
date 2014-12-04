from shapely.geometry import Point, MultiPoint, Polygon, GeometryCollection


def test_point():
    g = Point(0, 0)
    assert hash(g)


def test_multipoint():
    g = MultiPoint([(0, 0)])
    assert hash(g)


def test_polygon():
    g = Point(0, 0).buffer(1.0)
    assert hash(g)


def test_collection():
    g = GeometryCollection([Point(0, 0)])
    assert hash(g)
