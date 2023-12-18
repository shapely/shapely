
import numpy as np

from shapely._geoarrow import ArrayReader, GeoArrowGEOSException


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
