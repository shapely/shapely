import numpy as np

from shapely._geoarrow import ArrayBuilder, ArrayReader, GeoArrowGEOSException  # NOQA


def to_pyarrow(obj, schema_to):
    import pyarrow as pa

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
