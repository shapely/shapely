from shapely import GeometryCollection, LineString, MultiPoint, Point


def test_point():
    g = Point(1, 2)
    assert hash(g) == hash(Point(1, 2))
    assert hash(g) != hash(Point(1, 3))


def test_multipoint():
    g = MultiPoint([(1, 2), (3, 4)])
    assert hash(g) == hash(MultiPoint([(1, 2), (3, 4)]))
    assert hash(g) != hash(MultiPoint([(1, 2), (3, 3)]))


def test_linestring():
    g = LineString([(1, 2), (3, 4)])
    assert hash(g) == hash(LineString([(1, 2), (3, 4)]))
    assert hash(g) != hash(LineString([(1, 2), (3, 3)]))


def test_polygon():
    g = Point(0, 0).buffer(1.0)
    assert hash(g) == hash(Point(0, 0).buffer(1.0))
    assert hash(g) != hash(Point(0, 0).buffer(1.1))


def test_collection():
    g = GeometryCollection([Point(1, 2), LineString([(1, 2), (3, 4)])])
    assert hash(g) == hash(
        GeometryCollection([Point(1, 2), LineString([(1, 2), (3, 4)])])
    )
    assert hash(g) != hash(
        GeometryCollection([Point(1, 2), LineString([(1, 2), (3, 3)])])
    )
