import numpy as np
import pygeos
import pytest

from .common import line_string
from .common import point
from .common import point_z
from .common import all_types


def test_get_point():
    actual = pygeos.get_point(line_string, 1)
    assert pygeos.equals(actual, pygeos.points(1, 0))


def test_set_srid():
    actual = pygeos.set_srid(point, 4326)
    assert pygeos.get_srid(point) == 0
    assert pygeos.get_srid(actual) == 4326


def box_tpl(x1, y1, x2, y2):
    return (x2, y1), (x2, y2), (x1, y2), (x1, y1), (x2, y1)


def test_new_from_wkt():
    geom = point
    actual = pygeos.Geometry(geom.to_wkt())
    assert pygeos.equals(actual, geom)


def test_new_from_wkb():
    geom = point
    actual = pygeos.Geometry(geom.to_wkb())
    assert pygeos.equals(actual, geom)


def test_adapt_ptr_raises():
    geom = pygeos.clone(point)
    with pytest.raises(AttributeError):
        geom._ptr += 1


def test_to_wkt():
    assert point.to_wkt() == "POINT (2 2)"
    assert point.to_wkt(trim=False) == "POINT (2.000000 2.000000)"
    assert point.to_wkt(trim=False, precision=3) == "POINT (2.000 2.000)"
    assert point_z.to_wkt(dimension=2) == "POINT (1 1)"
    assert point_z.to_wkt(dimension=3) == "POINT Z (1 1 1)"
    assert point_z.to_wkt(dimension=3, use_old_3d=True) == "POINT (1 1 1)"


def test_to_wkb():
    be = b"\x00"
    le = b"\x01"
    point_type = b"\x01\x00\x00\x00"  # 1 as 32-bit uint (LE)
    point_type_3d = b"\x01\x00\x00\x80"
    coord = b"\x00\x00\x00\x00\x00\x00\xf0?"  # 1.0 as double (LE)

    assert point_z.to_wkb(dimension=2) == le + point_type + 2 * coord
    assert point_z.to_wkb(dimension=3) == le + point_type_3d + 3 * coord
    assert (
        point_z.to_wkb(dimension=2, byte_order=0)
        == be + point_type[::-1] + 2 * coord[::-1]
    )


def test_to_wkb_with_srid():
    point_with_srid = pygeos.set_srid(point, np.int32(4326))
    result = point_with_srid.to_wkb(include_srid=True)
    assert np.frombuffer(result[5:9], "<u4").item() == 4326


def test_to_wkb_hex():
    le = b"01"
    point_type = b"01000000"
    coord = b"000000000000F03F"  # 1.0 as double (LE)

    assert point_z.to_wkb(hex=True, dimension=2) == le + point_type + 2 * coord


@pytest.mark.parametrize("geom", all_types)
def test_from_wkt(geom):
    wkt = geom.to_wkt()
    actual = pygeos.Geometry.from_wkt(wkt)
    assert pygeos.equals(actual, geom)


def test_from_wkt_bytes():
    actual = pygeos.Geometry.from_wkt(b"POINT (2 2)")
    assert pygeos.equals(actual, point)


def test_from_wkt_exceptions():
    with pytest.raises(TypeError):
        pygeos.Geometry.from_wkt(list("POINT (2 2)"))
    with pytest.raises(TypeError):
        pygeos.Geometry.from_wkt(None)
    with pytest.raises(pygeos.GEOSException):
        pygeos.Geometry.from_wkt("")
    with pytest.raises(pygeos.GEOSException):
        pygeos.Geometry.from_wkt("NOT A WKT STRING")


@pytest.mark.parametrize("geom", all_types)
@pytest.mark.parametrize("use_hex", [False, True])
@pytest.mark.parametrize("byte_order", [0, 1])
def test_from_wkb(geom, use_hex, byte_order):
    wkb = geom.to_wkb(hex=use_hex, byte_order=byte_order)
    actual = pygeos.Geometry.from_wkb(wkb)
    assert pygeos.equals(actual, geom)


def test_from_wkb_typeerror():
    with pytest.raises(TypeError):
        pygeos.Geometry.from_wkb("\x01")
    with pytest.raises(TypeError):
        pygeos.Geometry.from_wkb(None)
    with pytest.raises(pygeos.GEOSException):
        pygeos.Geometry.from_wkb(b"POINT (2 2)")


def test_points_from_coords():
    actual = pygeos.points([[0, 0], [2, 2]])
    assert actual[0].to_wkt() == "POINT (0 0)"
    assert actual[1].to_wkt() == "POINT (2 2)"


def test_points_from_xy():
    actual = pygeos.points(2, [0, 1])
    assert actual[0].to_wkt() == "POINT (2 0)"
    assert actual[1].to_wkt() == "POINT (2 1)"


def test_points_from_xyz():
    actual = pygeos.points(1, 1, [0, 1])
    assert actual[0].to_wkt() == "POINT Z (1 1 0)"
    assert actual[1].to_wkt() == "POINT Z (1 1 1)"


def test_points_invalid_ndim():
    with pytest.raises(pygeos.GEOSException):
        pygeos.points([0, 1, 2, 3])


def test_linestrings_from_coords():
    actual = pygeos.linestrings([[[0, 0], [1, 1]], [[0, 0], [2, 2]]])
    assert actual[0].to_wkt() == "LINESTRING (0 0, 1 1)"
    assert actual[1].to_wkt() == "LINESTRING (0 0, 2 2)"


def test_linestrings_from_xy():
    actual = pygeos.linestrings([0, 1], [2, 3])
    assert actual.to_wkt() == "LINESTRING (0 2, 1 3)"


def test_linestrings_from_xy_broadcast():
    x = [0, 1]  # the same X coordinates for both linestrings
    y = [2, 3], [4, 5]  # each linestring has a different set of Y coordinates
    actual = pygeos.linestrings(x, y)
    assert actual[0].to_wkt() == "LINESTRING (0 2, 1 3)"
    assert actual[1].to_wkt() == "LINESTRING (0 4, 1 5)"


def test_linestrings_from_xyz():
    actual = pygeos.linestrings([0, 1], [2, 3], 0)
    assert actual.to_wkt() == "LINESTRING Z (0 2 0, 1 3 0)"


def test_linearrings():
    actual = pygeos.linearrings(box_tpl(0, 0, 1, 1))
    assert actual.to_wkt() == "LINEARRING (1 0, 1 1, 0 1, 0 0, 1 0)"


def test_linearrings_from_xy():
    actual = pygeos.linearrings([0, 1, 2, 0], [3, 4, 5, 3])
    assert actual.to_wkt() == "LINEARRING (0 3, 1 4, 2 5, 0 3)"


def test_linearrings_unclosed():
    actual = pygeos.linearrings(box_tpl(0, 0, 1, 1)[:-1])
    assert actual.to_wkt() == "LINEARRING (1 0, 1 1, 0 1, 0 0, 1 0)"


def test_polygon_from_linearring():
    actual = pygeos.polygons(pygeos.linearrings(box_tpl(0, 0, 1, 1)))
    assert actual.to_wkt() == "POLYGON ((1 0, 1 1, 0 1, 0 0, 1 0))"


def test_polygons():
    actual = pygeos.polygons(box_tpl(0, 0, 1, 1))
    assert actual.to_wkt() == "POLYGON ((1 0, 1 1, 0 1, 0 0, 1 0))"


def test_polygon_no_hole_list_raises():
    with pytest.raises(ValueError):
        pygeos.polygons(box_tpl(0, 0, 10, 10), box_tpl(1, 1, 2, 2))


def test_polygon_with_1_hole():
    actual = pygeos.polygons(box_tpl(0, 0, 10, 10), [box_tpl(1, 1, 2, 2)])
    assert pygeos.area(actual) == 99.0


def test_polygon_with_2_holes():
    actual = pygeos.polygons(
        box_tpl(0, 0, 10, 10), [box_tpl(1, 1, 2, 2), box_tpl(3, 3, 4, 4)]
    )
    assert pygeos.area(actual) == 98.0


def test_2_polygons_with_same_hole():
    actual = pygeos.polygons(
        [box_tpl(0, 0, 10, 10), box_tpl(0, 0, 5, 5)], [box_tpl(1, 1, 2, 2)]
    )
    assert pygeos.area(actual).tolist() == [99.0, 24.0]


def test_2_polygons_with_2_same_holes():
    actual = pygeos.polygons(
        [box_tpl(0, 0, 10, 10), box_tpl(0, 0, 5, 5)],
        [box_tpl(1, 1, 2, 2), box_tpl(3, 3, 4, 4)],
    )
    assert pygeos.area(actual).tolist() == [98.0, 23.0]


def test_2_polygons_with_different_holes():
    actual = pygeos.polygons(
        [box_tpl(0, 0, 10, 10), box_tpl(0, 0, 5, 5)],
        [[box_tpl(1, 1, 3, 3)], [box_tpl(1, 1, 2, 2)]],
    )
    assert pygeos.area(actual).tolist() == [96.0, 24.0]


def test_box():
    actual = pygeos.box(0, 0, 1, 1)
    assert actual.to_wkt() == "POLYGON ((1 0, 1 1, 0 1, 0 0, 1 0))"


def test_box_multiple():
    actual = pygeos.box(0, 0, [1, 2], [1, 2])
    assert actual[0].to_wkt() == "POLYGON ((1 0, 1 1, 0 1, 0 0, 1 0))"
    assert actual[1].to_wkt() == "POLYGON ((2 0, 2 2, 0 2, 0 0, 2 0))"
