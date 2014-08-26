from shapely.strtree import STRtree
from shapely.geometry import Point

def test_query():
    points = [Point(i, i) for i in range(10)]
    tree = STRtree(points)
    results = tree.query(Point(2,2).buffer(0.99))
    assert len(results) == 1
    results = tree.query(Point(2,2).buffer(1.0))
    assert len(results) == 3
