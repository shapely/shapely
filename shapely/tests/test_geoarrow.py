
import geoarrow.pyarrow as ga
from shapely.geoarrow import from_arrow

def test_from_arrow():
    from_arrow(None, ga.point())
