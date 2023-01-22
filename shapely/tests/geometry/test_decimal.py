from decimal import Decimal

import pytest

from shapely import (
    GeometryCollection,
    LinearRing,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)


def to_decimal(x):
    if isinstance(x, (int, float)):
        return Decimal(x)
    if isinstance(x, tuple):
        return tuple([to_decimal(x) for x in x])
    if isinstance(x, list):
        return list([to_decimal(x) for x in x])
    return x


def test_to_decimal():
    assert to_decimal(1) == Decimal(1)
    assert to_decimal([1]) == [Decimal(1)]
    assert to_decimal([1, 2]) == [Decimal(1), Decimal(2)]
    assert to_decimal((1, 2)) == (Decimal(1), Decimal(2))
    assert to_decimal([(1, 2), [3], 4, [5, 6, 7]]) == [
        (Decimal(1), Decimal(2)),
        [Decimal(3)],
        Decimal(4),
        [Decimal(5), Decimal(6), Decimal(7)],
    ]


items2d = [
    [(0.0, 0.0), (70.0, 120.0), (140.0, 0.0), (0.0, 0.0)],
    [(60.0, 80.0), (80.0, 80.0), (70.0, 60.0), (60.0, 80.0)],
    [(30.0, 10.0), (50.0, 10.0), (40.0, 30.0), (30.0, 10.0)],
    [(90.0, 10), (110.0, 10.0), (100.0, 30.0), (90.0, 10.0)],
]

items3d = [
    [(0.0, 0.0, 1.0), (70.0, 120.0, 2.0), (140.0, 0.0, 3.0), (0.0, 0.0, 1.0)],
    [(60.0, 80.0, 1.0), (80.0, 80.0, 2.0), (70.0, 60.0, 3.0), (60.0, 80.0, 1.0)],
    [(30.0, 10.0, 2.0), (50.0, 10.0, 3.0), (40.0, 30.0, 4.0), (30.0, 10.0, 2.0)],
    [(90.0, 10, 3.0), (110.0, 10.0, 4.0), (100.0, 30.0, 5.0), (90.0, 10.0, 6.0)],
]

all_geoms = [
    [
        Point(items[0][0]),
        Point(*items[0][0]),
        MultiPoint(items[0]),
        LinearRing(items[0]),
        LineString(items[0]),
        MultiLineString(items),
        Polygon(items[0]),
        MultiPolygon(
            [
                Polygon(items[2]),
                Polygon(items[0], holes=items[1:]),
                Polygon(items[0], holes=items[2:]),
            ]
        ),
        GeometryCollection([Point(items[0][0]), Polygon(items[0])]),
    ]
    for items in [items2d, to_decimal(items2d), items3d, to_decimal(items3d)]
]


@pytest.mark.parametrize("geoms", list(zip(*all_geoms)))
def test_decimal(geoms):
    assert geoms[0] == geoms[1]
    assert geoms[2] == geoms[3]
