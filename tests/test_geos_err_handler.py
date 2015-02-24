import logging

import pytest

from shapely.geometry import LineString
from shapely.geos import ReadingError
from shapely.wkt import loads


def test_error_handler(tmpdir):
    logger = logging.getLogger('shapely.geos')
    logger.setLevel(logging.DEBUG)

    logfile = str(tmpdir.join('test_error.log'))
    fh = logging.FileHandler(logfile)
    logger.addHandler(fh)

    # This operation calls error_handler with a format string that
    # has *no* conversion specifiers.
    LineString([(0, 0), (2, 2)]).project(LineString([(1, 1), (1.5, 1.5)]))
    
    # This calls error_handler with a format string of "%s" and one
    # value.
    with pytest.raises(ReadingError):
        loads('POINT (LOLWUT)')

    g = loads('MULTIPOLYGON (((10 20, 10 120, 60 70, 30 70, 30 40, 60 40, 60 70, 90 20, 10 20)))')
    assert g.is_valid == False

    log = open(logfile).read()
    assert "third argument of GEOSProject_r must be Point*" in log
    assert "Expected number but encountered word: 'LOLWUT'" in log
    assert "Ring Self-intersection at or near point 60 70" in log
