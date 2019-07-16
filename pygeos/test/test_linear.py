import pygeos
import numpy as np


def test_project():
    line = pygeos.linestrings([[0, 0], [1, 1], [2, 2]])
    points = pygeos.points([1, 3], [0, 3])
    actual = pygeos.project(line, points)
    expected = [0.5 * 2 ** 0.5, 2 * 2 ** 0.5]
    np.testing.assert_allclose(actual, expected)
