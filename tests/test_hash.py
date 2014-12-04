from shapely.geometry import Point


def test_hash():
    p = Point(0, 0)
    assert hash(p)
