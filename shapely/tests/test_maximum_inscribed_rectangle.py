import pytest
from shapely.geometry import Polygon
from shapely.algorithms import maximum_inscribed_rectangle


def test_simple_square():
    poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    rect = maximum_inscribed_rectangle(poly, resolution=20)
    assert rect.area == pytest.approx(100, rel=1e-3)


def test_concave():
    poly = Polygon([(0, 0), (10, 0), (10, 6), (6, 6), (6, 4), (0, 4)])
    rect = maximum_inscribed_rectangle(poly, resolution=20)
    assert rect.within(poly)


def test_thin_rectangle():
    poly = Polygon([(0, 0), (50, 0), (50, 2), (0, 2)])
    rect = maximum_inscribed_rectangle(poly, resolution=30)
    assert rect.area <= poly.area
    assert rect.within(poly)


def test_triangle():
    poly = Polygon([(0, 0), (10, 0), (5, 8)])
    rect = maximum_inscribed_rectangle(poly, resolution=40)
    assert rect.within(poly)
    assert rect.area > 0


def test_polygon_with_hole():
    outer = [(0, 0), (10, 0), (10, 10), (0, 10)]
    hole = [(3, 3), (7, 3), (7, 7), (3, 7)]
    poly = Polygon(outer, [hole])
    rect = maximum_inscribed_rectangle(poly, resolution=30)
    assert rect.within(poly)
    # Ensure it avoids the hole
    assert not rect.intersects(Polygon(hole))


def test_small_shape():
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    rect = maximum_inscribed_rectangle(poly, resolution=10)
    assert rect.within(poly)
    assert rect.area <= 1.0


def test_empty_polygon():
    poly = Polygon()
    rect = maximum_inscribed_rectangle(poly)
    assert rect.is_empty
