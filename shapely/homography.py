import numpy as np

import shapely

__all__ = ["perspective_transform"]


def perspective_transform(geom, matrix):
    r"""Return a transformed geometry using an perspective transformation (homography) matrix.

    The coefficient matrix is provided as a list or tuple or 1D numpy array with 8 items for 2D transformations
    (only supports 2D perspective transform currently).

    For 2D affine transformations, the 8 parameter matrix is::

        [a, b, c, d, e, f, xoff, yoff]

    which represents the augmented matrix:

        [x']   / a  b xoff \ [x]
        [y'] = | c  d yoff | [y]
        [1 ]   \ e  f   1  / [1]

    or the equations for the transformed coordinates::

        x' = (a * x + b * y + xoff) / (e * x + f * y + 1)
        y' = (c * x + d * y + yoff) / (e * x + f * y + 1)
    """
    if len(matrix) == 8:
        a, b, c, d, e, f, xoff, yoff = matrix
        if geom.has_z:
            raise ValueError("Only support 2D perspective transform for now")
    else:
        raise ValueError(
            "'matrix' expects 8 coefficients")

    A = np.array([[a, b], [c, d]], dtype=float)
    proj = np.array([e, f], dtype=float)
    off = np.array([xoff, yoff], dtype=float)

    def _affine_coords(coords):
        return (np.matmul(A, coords.T).T + off) / (np.matmul(proj, coords.T).T + 1)[..., None]

    return shapely.transform(geom, _affine_coords, include_z=False)
