import os
import doctest
from pathlib import Path

import pytest

optionflags = (doctest.REPORT_ONLY_FIRST_FAILURE |
               doctest.NORMALIZE_WHITESPACE |
               doctest.ELLIPSIS)


@pytest.mark.parametrize(
    "filename",
    list(Path(__file__).parent.glob('*.txt')),
)
def test_doctest(filename):
    ret = doctest.testfile(filename.name, optionflags=optionflags)
    assert ret.attempted > 0
    assert ret.failed == 0
