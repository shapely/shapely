import numpy as np

import shapely


def signed_area(ring):
    """Return the signed area enclosed by a ring in linear time using the
    algorithm at: https://web.archive.org/web/20080209143651/http://cgafaq.info:80/wiki/Polygon_Area
    """
    coords = np.array(ring.coords)[:, :2]
    xs, ys = np.vstack([coords, coords[1]]).T
    return np.sum(xs[1:-1] * (ys[2:] - ys[:-2])) / 2


def is_ccw_impl(name=None):
    """Predicate implementation"""
    return shapely.is_ccw
