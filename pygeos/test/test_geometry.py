import numpy as np
import pygeos
import pytest

from .common import point
from .common import line_string
from .common import linear_ring
from .common import polygon
from .common import polygon_with_hole
from .common import multi_point
from .common import multi_line_string
from .common import multi_polygon
from .common import geometry_collection
from .common import point_z
from .common import all_types


def test_get_num_points():
    actual = pygeos.get_num_points(all_types).tolist()
    assert actual == [0, 3, 5, 0, 0, 0, 0, 0, 0]


def test_get_num_interior_rings():
    actual = pygeos.get_num_interior_rings(all_types + (polygon_with_hole,)).tolist()
    assert actual == [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]


def test_get_num_geometries():
    actual = pygeos.get_num_geometries(all_types).tolist()
    assert actual == [1, 1, 1, 1, 2, 1, 2, 2, 0]


@pytest.mark.parametrize(
    "geom",
    [
        point,
        polygon,
        multi_point,
        multi_line_string,
        multi_polygon,
        geometry_collection,
    ],
)
def test_get_point_non_linestring(geom):
    actual = pygeos.get_point(geom, [0, 2, -1])
    assert pygeos.is_empty(actual).all()


@pytest.mark.parametrize("geom", [line_string, linear_ring])
def test_get_point(geom):
    n = pygeos.get_num_points(geom)
    actual = pygeos.get_point(geom, [0, -n, n, -(n + 1)])
    assert pygeos.equals(actual[0], actual[1]).all()
    assert pygeos.is_empty(actual[2:4]).all()


@pytest.mark.parametrize(
    "geom",
    [
        point,
        line_string,
        linear_ring,
        multi_point,
        multi_line_string,
        multi_polygon,
        geometry_collection,
    ],
)
def test_get_exterior_ring_non_polygon(geom):
    actual = pygeos.get_exterior_ring(geom)
    assert pygeos.is_empty(actual).all()


def test_get_exterior_ring():
    actual = pygeos.get_exterior_ring([polygon, polygon_with_hole])
    assert (pygeos.get_type_id(actual) == 2).all()


@pytest.mark.parametrize(
    "geom",
    [
        point,
        line_string,
        linear_ring,
        multi_point,
        multi_line_string,
        multi_polygon,
        geometry_collection,
    ],
)
def test_get_interior_ring_non_polygon(geom):
    actual = pygeos.get_interior_ring(geom, [0, 2, -1])
    assert pygeos.is_empty(actual).all()


def test_get_interior_ring():
    actual = pygeos.get_interior_ring(polygon_with_hole, [0, -1, 1, -2])
    assert pygeos.equals(actual[0], actual[1]).all()
    assert pygeos.is_empty(actual[2:4]).all()


@pytest.mark.parametrize("geom", [point, line_string, linear_ring, polygon])
def test_get_geometry_simple(geom):
    actual = pygeos.get_geometry(geom, [0, -1, 1, -2])
    assert pygeos.equals(actual[0], actual[1]).all()
    assert pygeos.is_empty(actual[2:4]).all()


@pytest.mark.parametrize(
    "geom", [multi_point, multi_line_string, multi_polygon, geometry_collection]
)
def test_get_geometry_collection(geom):
    n = pygeos.get_num_geometries(geom)
    actual = pygeos.get_geometry(geom, [0, -n, n, -(n + 1)])
    assert pygeos.equals(actual[0], actual[1]).all()
    assert pygeos.is_empty(actual[2:4]).all()


def test_get_type_id():
    assert pygeos.get_type_id(all_types).tolist()[:-1] == list(range(8))


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
    point = pygeos.Geometry.from_wkt("POINT (2 2)")
    with pytest.raises(AttributeError):
        point._ptr += 1


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


@pytest.mark.parametrize(
    "wkt", ("POINT EMPTY", "LINESTRING EMPTY", "GEOMETRYCOLLECTION EMPTY")
)
def test_from_wkt_empty(wkt):
    assert pygeos.Geometry.from_wkt(wkt) is pygeos.Empty


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
