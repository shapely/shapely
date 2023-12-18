
from shapely._geoarrow import ArrayReader


def from_arrow(arrays, schema):
    schema = schema.__arrow_c_schema__()
    reader = ArrayReader(schema)
