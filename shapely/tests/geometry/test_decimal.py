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


@pytest.mark.parametrize("items", [items2d, items3d])
def test_decimal(items):
    assert Point(items[0][0]) == Point(to_decimal(items[0][0]))
    assert Point(*items[0][0]) == Point(*to_decimal(items[0][0]))
    assert MultiPoint(items[0]) == MultiPoint(to_decimal(items[0]))
    assert LinearRing(items[0]) == LinearRing(to_decimal(items[0]))
    assert LineString(items[0]) == LineString(to_decimal(items[0]))
    assert MultiLineString(items) == MultiLineString(to_decimal(items))
    assert Polygon(items[0]) == Polygon(to_decimal(items[0]))
    assert Polygon(items[0], holes=items[1:]) == Polygon(
        to_decimal(items[0]), holes=to_decimal(items[1:])
    )
    assert MultiPolygon(
        [
            Polygon(items[2]),
            Polygon(items[0], holes=items[1:]),
            Polygon(items[0], holes=items[2:]),
        ]
    ) == MultiPolygon(
        [
            Polygon(to_decimal(items[2])),
            Polygon(to_decimal(items[0]), holes=to_decimal(items[1:])),
            Polygon(to_decimal(items[0]), holes=to_decimal(items[2:])),
        ]
    )
    assert GeometryCollection(
        [Point(items[0][0]), Polygon(items[0])]
    ) == GeometryCollection(
        [Point(to_decimal(items[0][0])), Polygon(to_decimal(items[0]))]
    )
