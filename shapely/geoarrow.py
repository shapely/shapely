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


class GeoArrowTypeExporter:

    def __init__(self, encoding, geometry_type=None, dimensions=None):
        encoding = Encoding(encoding).value
        wkb_type = 0

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
        elif geometry_type is not None:
            raise ValueError(f"Unknown value for geometry_type: '{geometry_type}'")

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
        else:
            raise ValueError(f"Unknown value for dimensions: '{dimensions}'")

        self._encoding = encoding
        self._wkb_type = wkb_type

    def __arrow_c_schema__(self):
        return SchemaCalculator.from_wkb_type(self._encoding, self._wkb_type)


class GeoArrowArrayExporter:

    def __init__(self, geometries, *, preferred_encoding=None) -> None:
        self._geometries = np.array(geometries)

        if preferred_encoding is None:
            self._preferred_encoding = Encoding.GEOARROW.value
        else:
            self._preferred_encoding = Encoding(preferred_encoding).value

    def __arrow_c_schema__(self):
        if self._preferred_encoding == Encoding.WKB.value:
            return SchemaCalculator.from_wkb_type(self._preferred_encoding)

        calculator = SchemaCalculator()
        calculator.ingest_geometry(self._geometries)
        return calculator.finish(self._preferred_encoding)

    def __arrow_c_array__(self, requested_schema=None):
        if requested_schema is None:
            requested_schema = self.__arrow_c_schema__()

        builder = ArrayBuilder(requested_schema)
        builder.append(self._geometries)
        chunks = builder.finish(ensure_non_empty=True)

        if len(chunks) == 1:
            return requested_schema, chunks[0]
        else:
            raise ValueError(
                f"Can't export geometries to single chunk ({len(chunks)} required)"
            )

    def __arrow_c_stream__(self, requested_schema=None):
        from nanoarrow.c_lib import CArrayStream, c_array, c_schema

        if requested_schema is None:
            requested_schema = self.__arrow_c_schema__()

        builder = ArrayBuilder(requested_schema)
        builder.append(self._geometries)
        chunks = builder.finish()

        na_schema = c_schema(requested_schema)
        na_chunks = [c_array(chunk) for chunk in chunks]
        na_stream = CArrayStream.from_array_list(
            na_chunks, na_schema, validate=False, move=True
        )
        return na_stream.__arrow_c_stream__()


def type_pyarrow(encoding, geometry_type=None, dimensions=None):
    import pyarrow as pa

    exporter = GeoArrowTypeExporter(encoding, geometry_type, dimensions)
    return pa.DataType._import_from_c_capsule(exporter.__arrow_c_schema__())


def infer_pyarrow_type(obj, encoding=None):
    import pyarrow as pa

    if encoding is None:
        encoding = Encoding.GEOARROW.value
    else:
        encoding = Encoding(encoding).value

    exporter = GeoArrowArrayExporter(obj, preferred_encoding=encoding)
    return pa.DataType._import_from_c_capsule(exporter.__arrow_c_schema__())


def to_pyarrow(obj, preferred_encoding=None):
    import pyarrow as pa

    exporter = GeoArrowArrayExporter(obj, preferred_encoding=preferred_encoding)
    return pa.array(exporter)


def from_arrow(arrays, schema, n=None):

    schema = schema.__arrow_c_schema__()
    reader = ArrayReader(schema)

    if n is not None:
        reader.reserve(n)

    for array in arrays:
        _, array = array.__arrow_c_array__()
        reader.read(array)

    return reader.finish()
