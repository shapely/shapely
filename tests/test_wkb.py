import binascii

import pytest

from shapely import wkt
from shapely.wkb import dumps, loads
from shapely.geometry import Point
from shapely.geos import geos_version



def bin2hex(value):
    return binascii.b2a_hex(value).upper().decode("utf-8")


def hex2bin(value):
    return binascii.a2b_hex(value)


def test_dumps_srid():
    p1 = Point(1.2, 3.4)
    result = dumps(p1)
    assert bin2hex(result) == "0101000000333333333333F33F3333333333330B40"
    result = dumps(p1, srid=4326)
    assert bin2hex(result) == "0101000020E6100000333333333333F33F3333333333330B40"


def test_dumps_endianness():
    p1 = Point(1.2, 3.4)
    result = dumps(p1)
    assert bin2hex(result) == "0101000000333333333333F33F3333333333330B40"
    result = dumps(p1, big_endian=False)
    assert bin2hex(result) == "0101000000333333333333F33F3333333333330B40"
    result = dumps(p1, big_endian=True)
    assert bin2hex(result) == "00000000013FF3333333333333400B333333333333"


def test_dumps_hex():
    p1 = Point(1.2, 3.4)
    result = dumps(p1, hex=True)
    assert result == "0101000000333333333333F33F3333333333330B40"
    

def test_loads_srid():
    # load a geometry which includes an srid
    geom = loads(hex2bin("0101000020E6100000333333333333F33F3333333333330B40"))
    assert isinstance(geom, Point)
    assert geom.coords[:] == [(1.2, 3.4)]
    # by default srid is not exported
    result = dumps(geom)
    assert bin2hex(result) == "0101000000333333333333F33F3333333333330B40"
    # include the srid in the output
    result = dumps(geom, include_srid=True)
    assert bin2hex(result) == "0101000020E6100000333333333333F33F3333333333330B40"
    # replace geometry srid with another
    result = dumps(geom, srid=27700)
    assert bin2hex(result) == "0101000020346C0000333333333333F33F3333333333330B40"


requires_geos_39 = pytest.mark.xfail(
    geos_version < (3, 9, 0), reason="GEOS >= 3.9.0 is required", strict=True)


@requires_geos_39
def test_point_empty():
    g = wkt.loads("POINT EMPTY")
    assert g.wkb_hex == "0101000000000000000000F87F000000000000F87F"


@requires_geos_39
def test_point_z_empty():
    g = wkt.loads("POINT Z EMPTY")
    assert g.wkb_hex == \
        "0101000080000000000000F87F000000000000F87F000000000000F87F"
