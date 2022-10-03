import pytest

from shapely import Point, Polygon


def test_format_point():

    # check basic cases
    pt = Point(1, 2, 3)
    assert f"{pt}" == pt.wkt
    assert format(pt, "") == pt.wkt
    assert format(pt, "x") == pt.wkb_hex.lower()
    assert format(pt, "X") == pt.wkb_hex

    # without specified precision, Python and GEOS' defaults are different
    assert (
        format(Point(1, 2, 3), "f")
        == "POINT Z (1.0000000000000000 2.0000000000000000 3.0000000000000000)"
    )
    assert format(Point(1, 2, 3), "g") == "POINT Z (1 2 3)"

    # check consistency with Python's float format and geomety format
    x1 = 0.12345678901234567
    y1 = 1.2345678901234567e10
    valid = [
        (".0f", x1, y1, "0", "12345678901"),
        (".1f", x1, y1, "0.1", "12345678901.2"),
        (".1g", x1, y1, "0.1", "1e+10"),
        (".6G", x1, y1, "0.123457", "1.23457E+10"),
        ("0.12g", x1, y1, "0.123456789012", "12345678901.2"),
        (".3F", float("inf"), -float("inf"), "INF", "-INF"),
    ]
    for format_spec, x, y, expected_x, expected_y in valid:
        assert format(x, format_spec) == expected_x, format_spec
        assert format(y, format_spec) == expected_y, format_spec
        expected_pt = f"POINT ({expected_x} {expected_y})"
        assert format(Point(x, y), format_spec) == expected_pt, format_spec

    invalid = [
        ("5G", ValueError, "invalid format specifier"),
        ("0.2e", ValueError, "invalid format specifier"),
        (".1x", ValueError, "hex representation does not specify precision"),
    ]
    for format_spec, err, match in invalid:
        with pytest.raises(err, match=match):
            format(pt, format_spec)


def test_format_polygon():
    # check basic cases
    poly = Point(0, 0).buffer(10, 2)
    assert f"{poly}" == poly.wkt
    assert format(poly, "") == poly.wkt
    assert format(poly, "x") == poly.wkb_hex.lower()
    assert format(poly, "X") == poly.wkb_hex

    # Use f-strings with extra characters and rounding precision
    assert f"<{poly:.2f}>" == (
        "<POLYGON ((10.00 0.00, 7.07 -7.07, 0.00 -10.00, -7.07 -7.07, "
        "-10.00 -0.00, -7.07 7.07, -0.00 10.00, 7.07 7.07, 10.00 0.00))>"
    )
    assert f"{poly:.2G}" == (
        "POLYGON ((10 0, 7.1 -7.1, 1.6E-14 -10, -7.1 -7.1, "
        "-10 -3.2E-14, -7.1 7.1, -4.6E-14 10, 7.1 7.1, 10 0))"
    )

    empty = Polygon()
    assert f"{empty}" == "POLYGON EMPTY"
    assert format(empty, "") == empty.wkt
    assert format(empty, ".2G") == empty.wkt
    assert format(empty, "x") == empty.wkb_hex.lower()
    assert format(empty, "X") == empty.wkb_hex
