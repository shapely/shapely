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


def test_to_pyarrow_empty():
    out = to_pyarrow([], ga.point())
    assert out == pa.chunked_array([], type=ga.point())


def test_to_pyarrow_wkt():
    out = to_pyarrow(
        [shapely.from_wkt("POINT (0 1)"), shapely.from_wkt("POINT (2 3)")], ga.wkt()
    )
    assert out == pa.chunked_array([ga.as_wkt(["POINT (0 1)", "POINT (2 3)"])])


def test_to_pyarrow_more_than_chunk_size():
    # 2 chunks plus one extra item
    chunk1 = [shapely.from_wkt("POINT (0 1)"), shapely.from_wkt("POINT (2 3)")] * 512
    chunk2 = [shapely.from_wkt("POINT (4 5)"), shapely.from_wkt("POINT (6 7)")] * 512
    chunk3 = [shapely.from_wkt("POINT (8 9)")]
    chunk_all = chunk1 + chunk2 + chunk3
    out = to_pyarrow(chunk_all, ga.wkt())

    assert len(out) == len(chunk_all)
    assert out.num_chunks == 1
    assert out.chunk(0)[0:2] == ga.as_wkt(["POINT (0 1)", "POINT (2 3)"])
    assert out.chunk(0)[1024:1026] == ga.as_wkt(["POINT (4 5)", "POINT (6 7)"])
    assert out.chunk(0)[2048:2049] == ga.as_wkt(["POINT (8 9)"])


def test_to_pyarrow_wkb():
    out = to_pyarrow(
        [shapely.from_wkt("POINT (0 1)"), shapely.from_wkt("POINT (2 3)")], ga.wkb()
    )
    assert out == pa.chunked_array([ga.as_wkb(["POINT (0 1)", "POINT (2 3)"])])


def test_to_pyarrow_point():
    out = to_pyarrow(
        [shapely.from_wkt("POINT (0 1)"), shapely.from_wkt("POINT (2 3)")], ga.point()
    )
    assert out == pa.chunked_array([ga.as_geoarrow(["POINT (0 1)", "POINT (2 3)"])])


def test_from_arrow_error_construct():
    with pytest.raises(GeoArrowGEOSException, match="Expected extension type"):
        from_arrow([], pa.float64())


def test_from_arrow_empty():
    testing.assert_array_equal(from_arrow([], ga.point()), np.array([], dtype=object))


def test_from_arrow_wkt():
    array = ga.as_wkt(["POINT (0 1)", "POINT (2 3)"])
    out = from_arrow([array], array.type)
    assert len(out) == 2
    assert out[0] == shapely.from_wkt("POINT (0 1)")
    assert out[1] == shapely.from_wkt("POINT (2 3)")
