import shapely.geometry


def test_feature_from_geo_interface():
    # https://github.com/shapely/shapely/issues/1814
    class Feature:
        @property
        def __geo_interface__(self):
            return {'type': "Feature", "geometry": {'type': "Point", "coordinates": [0, 0]}}


    expected = shapely.Point([0, 0])
    result = shapely.geometry.shape(Feature())
    assert result == expected
