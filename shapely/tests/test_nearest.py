from shapely.geometry import Point
from shapely.ops import nearest_points

def test_nearest():
    first, second = nearest_points(
                        Point(0, 0).buffer(1.0), Point(3, 0).buffer(1.0))
    assert round(first.x, 7) == 1.0
    assert round(second.x, 7) == 2.0
    assert round(first.y, 7) == round(second.y, 7) == 0.0
