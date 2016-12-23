# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the BSD License. See the LICENSE file in the root of this repository
# for complete details.
from __future__ import absolute_import, division, print_function

import pytest

from packaging.utils import canonicalize_name


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("foo", "foo"),
        ("Foo", "foo"),
        ("fOo", "foo"),
        ("foo.bar", "foo-bar"),
        ("Foo.Bar", "foo-bar"),
        ("Foo.....Bar", "foo-bar"),
        ("foo_bar", "foo-bar"),
        ("foo___bar", "foo-bar"),
        ("foo-bar", "foo-bar"),
        ("foo----bar", "foo-bar"),
    ],
)
def test_canonicalize_name(name, expected):
    assert canonicalize_name(name) == expected
