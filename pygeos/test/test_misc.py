import pygeos


def test_version():
    assert isinstance(pygeos.__version__, str)


def test_geos_version():
    assert isinstance(pygeos.geos_version, str)
