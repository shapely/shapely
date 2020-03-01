import pytest

from shapely.geometry import shape, Polygon


@pytest.mark.parametrize(
    "geom",
    [{"type": "Polygon", "coordinates": None}, {"type": "Polygon", "coordinates": []}],
)
def test_polygon_no_coords(geom):
    assert shape(geom) == Polygon()


def test_polygon_empty_np_array():
    np = pytest.importorskip("numpy")
    geom = {"type": "Polygon", "coordinates": np.array([])}
    assert shape(geom) == Polygon()


def test_polygon_with_coords_list():
    geom = {"type": "Polygon", "coordinates": [[[5, 10], [10, 10], [10, 5]]]}
    obj = shape(geom)
    assert obj == Polygon([(5, 10), (10, 10), (10, 5)])


def test_polygon_not_empty_np_array():
    np = pytest.importorskip("numpy")
    geom = {"type": "Polygon", "coordinates": np.array([[[5, 10], [10, 10], [10, 5]]])}
    obj = shape(geom)
    assert obj == Polygon([(5, 10), (10, 10), (10, 5)])
