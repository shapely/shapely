import numpy as np

import shapely

__all__ = ["perspective_transform"]


def perspective_transform(geom, matrix):
    r"""Return a transformed geometry using an perspective transformation (homography) matrix.
    The coefficient matrix is provided as a list or tuple or 1D numpy array with 8 items,
    or a 2D numpy array with 3x3 shape (like the output of cv.findHomography).
    for 2D transformations.
    For 2D affine transformations, the 8 parameter matrix is::
        [a, b, c, d, e, f, xoff, yoff]
    which represents the augmented 3x3 matrix:
        [s*x']   / a  b xoff \ [x]
        [s*y'] = | c  d yoff | [y]
        [s   ]   \ e  f   1  / [1]
    or the equations for the transformed coordinates::
        x' = (a * x + b * y + xoff) / (e * x + f * y + 1)
        y' = (c * x + d * y + yoff) / (e * x + f * y + 1)
    """
    if isinstance(matrix, np.ndarray) and matrix.shape == (3, 3):
        a = matrix[0][0]
        b = matrix[0][1]
        c = matrix[1][0]
        d = matrix[1][1]
        e = matrix[2][0]
        f = matrix[2][1]
        xoff = matrix[0][2]
        yoff = matrix[1][2]

    elif len(matrix) == 8:
        a, b, c, d, e, f, xoff, yoff = matrix

    else:
        raise ValueError(
            "'matrix' expects either 8 coefficients in a list/tuple/1d array"
            "or 3x3 array")

    A = np.array([[a, b], [c, d]], dtype=float)
    proj = np.array([e, f], dtype=float)
    off = np.array([xoff, yoff], dtype=float)

    def _affine_coords(coords):
        return (np.matmul(A, coords.T).T + off) / (np.matmul(proj, coords.T).T + 1)[..., None]

    return shapely.transform(geom, _affine_coords, include_z=False)
