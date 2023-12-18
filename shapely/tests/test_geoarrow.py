import geoarrow.pyarrow as ga
import numpy as np
import pyarrow as pa
import pytest
from numpy import testing

import shapely
from shapely.geoarrow import from_arrow, GeoArrowGEOSException


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
