import pygeos


def test_version():
    assert isinstance(pygeos.__version__, str)


def test_geos_version():
    expected = "{0:d}.{1:d}.{2:d}".format(*pygeos.geos_version)
    assert pygeos.geos_version_string == expected


def test_geos_capi_version():
    expected = "{0:d}.{1:d}.{2:d}-CAPI-{3:d}.{4:d}.{5:d}".format(
        *(pygeos.geos_version + pygeos.geos_capi_version)
    )
    assert pygeos.geos_capi_version_string == expected
