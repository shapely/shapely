import geoarrow.pyarrow as ga
import numpy as np
import pyarrow as pa
import pytest
from numpy import testing

import shapely
from shapely.geoarrow import (
    Encoding,
    type_pyarrow,
    to_pyarrow,
    from_arrow,
    infer_pyarrow_type,
    GeoArrowGEOSException,
)


def test_type_pyarrow():
    assert type_pyarrow(Encoding.WKB) == ga.wkb()
    assert type_pyarrow(Encoding.WKT) == ga.wkt()

    assert type_pyarrow(Encoding.GEOARROW, shapely.GeometryType.POINT) == ga.point()
    assert (
        type_pyarrow(Encoding.GEOARROW, shapely.GeometryType.LINESTRING)
        == ga.linestring()
    )
    assert type_pyarrow(Encoding.GEOARROW, shapely.GeometryType.POLYGON) == ga.polygon()
    assert (
        type_pyarrow(Encoding.GEOARROW, shapely.GeometryType.MULTIPOINT)
        == ga.multipoint()
    )
    assert (
        type_pyarrow(Encoding.GEOARROW, shapely.GeometryType.MULTILINESTRING)
        == ga.multilinestring()
    )
    assert (
        type_pyarrow(Encoding.GEOARROW, shapely.GeometryType.MULTIPOLYGON)
        == ga.multipolygon()
    )

    assert type_pyarrow(
        Encoding.GEOARROW_INTERLEAVED, shapely.GeometryType.POINT
    ) == ga.point().with_coord_type(ga.CoordType.INTERLEAVED)

    assert type_pyarrow(
        Encoding.GEOARROW, shapely.GeometryType.POINT, "xyz"
    ) == ga.point().with_dimensions(ga.Dimensions.XYZ)

    assert type_pyarrow(
        Encoding.GEOARROW, shapely.GeometryType.POINT, "xym"
    ) == ga.point().with_dimensions(ga.Dimensions.XYM)

    assert type_pyarrow(
        Encoding.GEOARROW, shapely.GeometryType.POINT, "xyzm"
    ) == ga.point().with_dimensions(ga.Dimensions.XYZM)


def test_infer_pyarrow_type():
    type = infer_pyarrow_type(
        [shapely.from_wkt("POINT (0 1)"), shapely.from_wkt("POINT (2 3)")]
    )
    assert type == ga.point()

    type = infer_pyarrow_type(
        [shapely.from_wkt("POINT (0 1)"), shapely.from_wkt("POINT (2 3)")],
        encoding=Encoding.GEOARROW_INTERLEAVED,
    )
    assert type == ga.point().with_coord_type(ga.CoordType.INTERLEAVED)

    type = infer_pyarrow_type(
        [shapely.from_wkt("POINT (0 1)"), shapely.from_wkt("MULTIPOINT (2 3)")]
    )
    assert type == ga.multipoint()

    type = infer_pyarrow_type(
        [shapely.from_wkt("POINT (0 1)"), shapely.from_wkt("LINESTRING (2 3, 4 5)")]
    )
    assert type == ga.wkb()


def test_infer_pyarrow_type_more_than_chunk_size():
    chunk1 = [shapely.from_wkt("POINT (0 1)"), shapely.from_wkt("POINT (2 3)")] * 512
    chunk2 = [shapely.from_wkt("LINESTRING (0 1, 2 3)")]

    type = infer_pyarrow_type(chunk1 + chunk2)
    assert type == ga.wkb()


def test_to_pyarrow_empty():
    out = to_pyarrow([])
    assert out == pa.array([], type=ga.wkb())


def test_to_pyarrow_wkt():
    out = to_pyarrow(
        [shapely.from_wkt("POINT (0 1)"), shapely.from_wkt("POINT (2 3)")], Encoding.WKT
    )
    assert out == ga.as_wkt(["POINT (0 1)", "POINT (2 3)"])


def test_to_pyarrow_more_than_chunk_size():
    # 2 chunks plus one extra item
    chunk1 = [shapely.from_wkt("POINT (0 1)"), shapely.from_wkt("POINT (2 3)")] * 512
    chunk2 = [shapely.from_wkt("POINT (4 5)"), shapely.from_wkt("POINT (6 7)")] * 512
    chunk3 = [shapely.from_wkt("POINT (8 9)")]
    chunk_all = chunk1 + chunk2 + chunk3
    out = to_pyarrow(chunk_all, Encoding.WKT)

    assert out[0:2] == ga.as_wkt(["POINT (0 1)", "POINT (2 3)"])
    assert out[1024:1026] == ga.as_wkt(["POINT (4 5)", "POINT (6 7)"])
    assert out[2048:2049] == ga.as_wkt(["POINT (8 9)"])


def test_to_pyarrow_wkb():
    out = to_pyarrow(
        [shapely.from_wkt("POINT (0 1)"), shapely.from_wkt("POINT (2 3)")], Encoding.WKB
    )
    assert out == ga.as_wkb(["POINT (0 1)", "POINT (2 3)"])


def test_to_pyarrow_point():
    out = to_pyarrow(
        [shapely.from_wkt("POINT (0 1)"), shapely.from_wkt("POINT (2 3)")],
        Encoding.GEOARROW,
    )
    assert out == ga.as_geoarrow(["POINT (0 1)", "POINT (2 3)"])


def test_from_arrow_error_construct():
    with pytest.raises(GeoArrowGEOSException, match="Expected extension type"):
        from_arrow(pa.array([], pa.float64()))


def test_from_arrow_empty():
    testing.assert_array_equal(from_arrow(pa.array([], ga.point())), np.array([], dtype=object))


def test_from_arrow_wkt():
    array = ga.as_wkt(["POINT (0 1)", "POINT (2 3)"])
    out = from_arrow(array)
    assert len(out) == 2
    assert out[0] == shapely.from_wkt("POINT (0 1)")
    assert out[1] == shapely.from_wkt("POINT (2 3)")

def test_from_arrow_chunked_array():
    array = ga.as_wkt(["POINT (0 1)", "POINT (2 3)"])
    chunked_array = pa.chunked_array([array])
    out = from_arrow(chunked_array)
    assert len(out) == 2
    assert out[0] == shapely.from_wkt("POINT (0 1)")
    assert out[1] == shapely.from_wkt("POINT (2 3)")
