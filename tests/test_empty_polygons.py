from shapely.geometry import MultiPolygon, Polygon


def test_empty_polygon():
    """No constructor arg makes an empty polygon geometry."""
    assert Polygon().is_empty


def test_empty_multipolygon():
    """No constructor arg makes an empty multipolygon geometry."""
    assert MultiPolygon().is_empty


def test_multipolygon_empty_polygon():
    """An empty polygon passed to MultiPolygon() makes an empty
    multipolygon geometry."""
    assert MultiPolygon([Polygon()]).is_empty
