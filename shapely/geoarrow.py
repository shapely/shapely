import enum

import numpy as np

from shapely import GeometryType
from shapely._geoarrow import (
    SchemaCalculator,
    ArrayBuilder,
    ArrayReader,
    GeoArrowGEOSException,
)  # NOQA


class Encoding(enum.Enum):
    WKB = SchemaCalculator.ENCODING_WKB
    WKT = SchemaCalculator.ENCODING_WKT
    GEOARROW = SchemaCalculator.ENCODING_GEOARROW
    GEOARROW_INTERLEAVED = SchemaCalculator.ENCODING_GEOARROW_INTERLEAVED


def type_pyarrow(encoding, geometry_type=None, dimensions=None):
    import pyarrow as pa

    encoding = Encoding(encoding).value

    if geometry_type == GeometryType.POINT:
        wkb_type = 1
    elif geometry_type in (GeometryType.LINESTRING, GeometryType.LINEARRING):
        wkb_type = 2
    elif geometry_type == GeometryType.POLYGON:
        wkb_type = 3
    elif geometry_type == GeometryType.MULTIPOINT:
        wkb_type = 4
    elif geometry_type == GeometryType.MULTILINESTRING:
        wkb_type = 5
    elif geometry_type == GeometryType.MULTIPOLYGON:
        wkb_type = 6
    elif geometry_type == GeometryType.GEOMETRYCOLLECTION:
        wkb_type = 7
    else:
        wkb_type = 0

    if dimensions is not None:
        dimensions = dimensions.lower()

    if dimensions == "xy" or dimensions is None:
        wkb_type += 0
    elif dimensions == "xyz":
        wkb_type += 1000
    elif dimensions == "xym":
        wkb_type += 2000
    elif dimensions == "xyzm":
        wkb_type += 3000

    holder = SchemaCalculator.from_wkb_type(encoding, wkb_type)
    return pa.DataType._import_from_c(holder._addr())


def infer_pyarrow_type(obj, encoding=None):
    import pyarrow as pa

    if encoding is None:
        encoding = Encoding.GEOARROW.value
    else:
        encoding = Encoding(encoding).value

    # Note: faster to iterate over np.array(obj)
    obj = np.array(obj)

    calculator = SchemaCalculator()
    calculator.ingest_geometry(obj)
    holder = calculator.finish(encoding)

    return pa.DataType._import_from_c(holder._addr())


def to_pyarrow(obj, schema_to):
    import pyarrow as pa

    # Note: faster to iterate over np.array(obj)
    obj = np.array(obj)

    builder = ArrayBuilder(schema_to.__arrow_c_schema__())
    builder.append(obj)

    chunks_pyarrow = []
    for holder in builder.finish():
        array = pa.Array._import_from_c(holder._addr(), schema_to)
        chunks_pyarrow.append(array)

    return pa.chunked_array(chunks_pyarrow, type=schema_to)


def from_arrow(arrays, schema):
    schema = schema.__arrow_c_schema__()
    reader = ArrayReader(schema)
    geom_arrays = []
    for array in arrays:
        _, array = array.__arrow_c_array__()
        geom_arrays.append(reader.read(array))

    if geom_arrays:
        return np.concatenate(geom_arrays)
    else:
        return np.array([], dtype=object)
