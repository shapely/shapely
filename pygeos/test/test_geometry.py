import numpy as np
import pygeos
import pytest

from .common import line_string
from .common import point
from .common import point_z
from .common import all_types


def test_get_type_id():
    assert pygeos.get_type_id(all_types).tolist() == list(range(8))


def test_get_num_points():
    assert pygeos.get_num_points(line_string) == 3


def test_get_point():
    actual = pygeos.get_point(line_string, 1)
    assert pygeos.equals(actual, pygeos.points(1, 0))


def test_get_set_srid():
    actual = pygeos.set_srid(point, 4326)
    assert pygeos.get_srid(point) == 0
    assert pygeos.get_srid(actual) == 4326


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
