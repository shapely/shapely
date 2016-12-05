"""Affine transforms, both in general and specific, named transforms."""

from math import sin, cos, tan, pi
from collections import namedtuple


__all__ = ['affine_transform', 'rotate', 'scale', 'skew', 'translate']


class affine_matrix_builder:
    """
    Class for building affine transformation matrices.
    Reduces the number of transformations needed by combining the
    affine matrices when appropriate.
    Class implemented by William Rusnack, github.com/BebeSparkelSparkel, williamrusnack@gmail.com
    """
    def __init__(self, geom, matrix=None):
        self.geom = geom

        self.last_transform = None
        self.current_transform = None

        self.matrix = None
        if matrix is not None: self.affine_transform(matrix)

    def affine_transform(self, matrix):
        """
        Applies a transforme matrix using an affine transformation matrix.

        The coefficient matrix is provided as a list or tuple with 6 or 12 items
        for 2D or 3D transformations, respectively.

        For 2D affine transformations, the 6 parameter matrix is::

            [a, b, d, e, xoff, yoff]

        which represents the augmented matrix::

                                / a  b xoff \
            [x' y' 1] = [x y 1] | d  e yoff |
                                \ 0  0   1  /

        or the equations for the transformed coordinates::

            x' = a * x + b * y + xoff
            y' = d * x + e * y + yoff

        For 3D affine transformations, the 12 parameter matrix is::

            [a, b, c, d, e, f, g, h, i, xoff, yoff, zoff]

        which represents the augmented matrix::

                                     / a  b  c xoff \
            [x' y' z' 1] = [x y z 1] | d  e  f yoff |
                                     | g  h  i zoff |
                                     \ 0  0  0   1  /

        or the equations for the transformed coordinates::

            x' = a * x + b * y + c * z + xoff
            y' = d * x + e * y + f * z + yoff
            z' = g * x + h * y + i * z + zoff
        """
        if self.current_transform is None:
            self.current_transform = self._transform_log(self.affine_transform, {'matrix': matrix})

        if self.matrix is None:
            self.matrix = matrix
        else:
            if self._combine_matrices():
                num_elem = 9
                matrix_size = int(num_elem**.5)

                self.matrix = multiply_matrices(self.matrix[:num_elem], matrix[:num_elem]) + \
                    [e1 + e2 for e1, e2 in zip(self.matrix[num_elem:], matrix[num_elem:])]

            else:
                self.geom = self.transform()
                self.matrix = matrix

        self.last_transform = self.current_transform
        self.current_transform = None

        return self

    def rotate(self, angle, origin='center', use_radians=False):
        """
        Applies a rotate matrix on a 2D plane.

        The angle of rotation can be specified in either degrees (default) or
        radians by setting ``use_radians=True``. Positive angles are
        counter-clockwise and negative are clockwise rotations.

        The point of origin can be a keyword 'center' for the bounding box
        center (default), 'centroid' for the geometry's centroid, a Point object
        or a coordinate tuple (x0, y0).

        The affine transformation matrix for 2D rotation is:

          / cos(r) -sin(r) xoff \
          | sin(r)  cos(r) yoff |
          \   0       0      1  /

        where the offsets are calculated from the origin Point(x0, y0):

            xoff = x0 - x0 * cos(r) + y0 * sin(r)
            yoff = y0 - x0 * sin(r) - y0 * cos(r)
        """
        self.current_transform = self._transform_log(self.rotate, {'angle': angle, 'origin': origin, 'use_radians': use_radians})

        if not use_radians:  # convert from degrees
            angle *= pi/180.0
        cosp = cos(angle)
        sinp = sin(angle)
        if abs(cosp) < 2.5e-16:
            cosp = 0.0
        if abs(sinp) < 2.5e-16:
            sinp = 0.0
        x0, y0 = self._interpret_origin(self.geom, origin, 2)

        matrix = (cosp, -sinp, 0.0,
                  sinp,  cosp, 0.0,
                  0.0,    0.0, 1.0,
                  x0 - x0 * cosp + y0 * sinp, y0 - x0 * sinp - y0 * cosp, 0.0)

        # if self.matrix is None: self.matrix = matrix
        # else: self.matrix = self.matmult(self.matrix, matrix)
        # return self

        return self.affine_transform(matrix)

    def scale(self, xfact=1.0, yfact=1.0, zfact=1.0, origin='center'):
        """
        Applies a scale matrix, scaled by factors along each dimension.

        The point of origin can be a keyword 'center' for the 2D bounding box
        center (default), 'centroid' for the geometry's 2D centroid, a Point
        object or a coordinate tuple (x0, y0, z0).

        Negative scale factors will mirror or reflect coordinates.

        The general 3D affine transformation matrix for scaling is:

            / xfact  0    0   xoff \
            |   0  yfact  0   yoff |
            |   0    0  zfact zoff |
            \   0    0    0     1  /

        where the offsets are calculated from the origin Point(x0, y0, z0):

            xoff = x0 - x0 * xfact
            yoff = y0 - y0 * yfact
            zoff = z0 - z0 * zfact
        """
        self.current_transform = self._transform_log(self.scale, {'xfact': xfact, 'yfact': yfact, 'zfact': zfact, 'origin': origin})

        x0, y0, z0 = self._interpret_origin(self.geom, origin, 3)

        matrix = (xfact, 0.0, 0.0,
                  0.0, yfact, 0.0,
                  0.0, 0.0, zfact,
                  x0 - x0 * xfact, y0 - y0 * yfact, z0 - z0 * zfact)

        # if self.matrix is None: self.matrix = matrix
        # else: self.matrix = self.matmult(self.matrix, matrix)
        # return self

        return self.affine_transform(matrix)

    def skew(self, xs=0.0, ys=0.0, origin='center', use_radians=False):
        """
        Applies a skew matrix, sheared by angles along x and y dimensions.

        The shear angle can be specified in either degrees (default) or radians
        by setting ``use_radians=True``.

        The point of origin can be a keyword 'center' for the bounding box
        center (default), 'centroid' for the geometry's centroid, a Point object
        or a coordinate tuple (x0, y0).

        The general 2D affine transformation matrix for skewing is:

            /   1    tan(xs) xoff \
            | tan(ys)  1     yoff |
            \   0      0       1  /

        where the offsets are calculated from the origin Point(x0, y0):

            xoff = -y0 * tan(xs)
            yoff = -x0 * tan(ys)
        """
        self.current_transform = self._transform_log(self.skew, {'xs': xs, 'ys': ys, 'origin': origin, 'use_radians': use_radians})

        if not use_radians:  # convert from degrees
            xs *= pi/180.0
            ys *= pi/180.0
        tanx = tan(xs)
        tany = tan(ys)
        if abs(tanx) < 2.5e-16:
            tanx = 0.0
        if abs(tany) < 2.5e-16:
            tany = 0.0
        x0, y0 = self._interpret_origin(self.geom, origin, 2)

        matrix = (1.0, tanx, 0.0,
                  tany, 1.0, 0.0,
                  0.0,  0.0, 1.0,
                  -y0 * tanx, -x0 * tany, 0.0)

        # if self.matrix is None: self.matrix = matrix
        # else: self.matrix = self.matmult(self.matrix, matrix)
        # return self

        return self.affine_transform(matrix)

    def translate(self, xoff=0.0, yoff=0.0, zoff=0.0):
        """
        Applies a translate matrix that shifts by offsets along each dimension.

        The general 3D affine transformation matrix for translation is:

            / 1  0  0 xoff \
            | 0  1  0 yoff |
            | 0  0  1 zoff |
            \ 0  0  0   1  /
        """
        self.current_transform = self._transform_log(self.translate, {'xoff': xoff, 'yoff': yoff, 'zoff': zoff})

        matrix = (1.0, 0.0, 0.0,
                  0.0, 1.0, 0.0,
                  0.0, 0.0, 1.0,
                  xoff, yoff, zoff)

        # if self.matrix is None: self.matrix = matrix
        # else: self.matrix = self.matmult(self.matrix, matrix)
        # return self

        return self.affine_transform(matrix)

    def transform(self):
        """Returns a transformed geometry using an affine transformation matrix.

        The coefficient matrix is provided as a list or tuple with 6 or 12 items
        for 2D or 3D transformations, respectively.

        For 2D affine transformations, the 6 parameter matrix is::

            [a, b, d, e, xoff, yoff]

        which represents the augmented matrix::

                                / a  b xoff \
            [x' y' 1] = [x y 1] | d  e yoff |
                                \ 0  0   1  /

        or the equations for the transformed coordinates::

            x' = a * x + b * y + xoff
            y' = d * x + e * y + yoff

        For 3D affine transformations, the 12 parameter matrix is::

            [a, b, c, d, e, f, g, h, i, xoff, yoff, zoff]

        which represents the augmented matrix::

                                     / a  b  c xoff \
            [x' y' z' 1] = [x y z 1] | d  e  f yoff |
                                     | g  h  i zoff |
                                     \ 0  0  0   1  /

        or the equations for the transformed coordinates::

            x' = a * x + b * y + c * z + xoff
            y' = d * x + e * y + f * z + yoff
            z' = g * x + h * y + i * z + zoff
        """
        if self.matrix is None:
            raise ValueError('Affine matrix not defined.')

        if self.geom.is_empty:
            return self.geom
        if len(self.matrix) == 6:
            ndim = 2
            a, b, d, e, xoff, yoff = self.matrix
            if self.geom.has_z:
                ndim = 3
                i = 1.0
                c = f = g = h = zoff = 0.0
                self.matrix = a, b, c, d, e, f, g, h, i, xoff, yoff, zoff
        elif len(self.matrix) == 12:
            ndim = 3
            a, b, c, d, e, f, g, h, i, xoff, yoff, zoff = self.matrix
            if not self.geom.has_z:
                ndim = 2
                self.matrix = a, b, d, e, xoff, yoff
        else:
            raise ValueError("'matrix' expects either 6 or 12 coefficients")

        def affine_pts(pts):
            """Internal function to yield affine transform of coordinate tuples"""
            if ndim == 2:
                for x, y in pts:
                    xp = a * x + b * y + xoff
                    yp = d * x + e * y + yoff
                    yield (xp, yp)
            elif ndim == 3:
                for x, y, z in pts:
                    xp = a * x + b * y + c * z + xoff
                    yp = d * x + e * y + f * z + yoff
                    zp = g * x + h * y + i * z + zoff
                    yield (xp, yp, zp)

        # Process coordinates from each supported geometry type
        if self.geom.type in ('Point', 'LineString', 'LinearRing'):
            return type(self.geom)(list(affine_pts(self.geom.coords)))
        elif self.geom.type == 'Polygon':
            ring = self.geom.exterior
            shell = type(ring)(list(affine_pts(ring.coords)))
            holes = list(self.geom.interiors)
            for pos, ring in enumerate(holes):
                holes[pos] = type(ring)(list(affine_pts(ring.coords)))
            return type(self.geom)(shell, holes)
        elif self.geom.type.startswith('Multi') or self.geom.type == 'GeometryCollection':
            # Recursive call
            # TODO: fix GeometryCollection constructor
            return type(self.geom)([affine_transform(part, self.matrix)
                               for part in self.geom.geoms])
        else:
            raise ValueError('Type %r not recognized' % self.geom.type)

    _transform_log = namedtuple('_transform_log', ('transform', 'inputs'))

    def _combine_matrices(self):
        """
        Returns true if the affine if the last and current matrices can
        be safely combined to get the expected output.
        """
        # this can be optimized more to look at inputs/matrix too but has not been done yet
        return (self.last_transform.transform, self.current_transform.transform) \
            in self._compatable_combinations

    @staticmethod
    def _interpret_origin(geom, origin, ndim):
        """Returns interpreted coordinate tuple for origin parameter.

        This is a helper function for other transform functions.

        The point of origin can be a keyword 'center' for the 2D bounding box
        center, 'centroid' for the geometry's 2D centroid, a Point object or a
        coordinate tuple (x0, y0, z0).
        """
        # get coordinate tuple from 'origin' from keyword or Point type
        if origin == 'center':
            # bounding box center
            minx, miny, maxx, maxy = geom.bounds
            origin = ((maxx + minx)/2.0, (maxy + miny)/2.0)
        elif origin == 'centroid':
            origin = geom.centroid.coords[0]
        elif isinstance(origin, str):
            raise ValueError("'origin' keyword %r is not recognized" % origin)
        elif hasattr(origin, 'type') and origin.type == 'Point':
            origin = origin.coords[0]

        # origin should now be tuple-like
        if len(origin) not in (2, 3):
            raise ValueError("Expected number of items in 'origin' to be "
                             "either 2 or 3")
        if ndim == 2:
            return origin[0:2]
        else:  # 3D coordinate
            if len(origin) == 2:
                return origin + (0.0,)
            else:
                return origin

# Holds the safe relationships of when two affine matricies can be combined.
affine_matrix_builder._compatable_combinations = {
    (affine_matrix_builder.affine_transform,    affine_matrix_builder.translate),
    (affine_matrix_builder.rotate,              affine_matrix_builder.rotate),
    (affine_matrix_builder.rotate,              affine_matrix_builder.translate),
    (affine_matrix_builder.scale,               affine_matrix_builder.scale),
    (affine_matrix_builder.scale,               affine_matrix_builder.translate),
    (affine_matrix_builder.skew,                affine_matrix_builder.translate),
    (affine_matrix_builder.translate,           affine_matrix_builder.translate)
}


def square_matrix_size(matrix):
    size = len(matrix)**.5
    if size != int(size):
        ValueError('Not a square matrix')
    return int(size)

def row_iter(matrix):
    square_size = square_matrix_size(matrix)

    for i in range(0, len(matrix), square_size):
        yield matrix[i:i + square_size]

def get_column(matrix, column_index):
    return (row[column_index] for row in row_iter(matrix))

def column_iter(matrix):
    square_size = square_matrix_size(matrix)

    for i in range(square_size):
        yield get_column(matrix, i)

def multiply_matrices(a, b):
    result = []
    for row in row_iter(a):
        for col in column_iter(b):
            result.append(sum(r*c for r, c in zip(row, col)))
    return result



# for backwards compatability
def affine_transform(geom, matrix):
    return affine_matrix_builder(geom, matrix).transform()

def rotate(geom, angle, origin='center', use_radians=False):
    return affine_matrix_builder(geom).rotate(angle, origin, use_radians).transform()# def rotate(geom, angle, origin='center', use_radians=False):

def scale(geom, xfact=1.0, yfact=1.0, zfact=1.0, origin='center'):
    return affine_matrix_builder(geom).scale(xfact, yfact, zfact, origin).transform()

def skew(geom, xs=0.0, ys=0.0, origin='center', use_radians=False):
    return affine_matrix_builder(geom).skew(xs, ys, origin, use_radians).transform()

def translate(geom, xoff=0.0, yoff=0.0, zoff=0.0):
    return affine_matrix_builder(geom).translate(xoff, yoff, zoff).transform()


# # code used to determine how affine matrices can be combined
# from itertools import permutations, chain

# from shapely.geometry import Point
# from shapely.affinity import affine_matrix_builder as amb

# from pprint import pprint

# transform_functions = {
#     amb.affine_transform: {
#         'first': ((1,2,3,4,5,6,7,8,9,10,11,12),),
#         'different': ((2,3,4,5,6,7,8,9,10,11,12,13),),
#         'same_affine_matrix':((1,2,3,4,5,6,7,8,.9,.10,.11,.12),)
#     },
#     amb.rotate: {
#         'first': (90, (0,0)),
#         'different': (55, (.5, 7)),
#         'same_affine_matrix': (90, (.5, 7))
#     },
#     amb.scale: {
#         'first': (2, 3, 4, (0,0)),
#         'different': (4, 2, 3, (.5, 7)),
#         'same_affine_matrix': (2, 3, 4, (.5, 7))
#     },
#     amb.skew: {
#         'first': (2, 3, (0,0)),
#         'different': (4, 5, (.5, 7)),
#         'same_affine_matrix': (2, 3, (.5, 7))
#     },
#     amb.translate: {
#         'first': (1, 2, 3),
#         'different': (2, 3, 4),
#         'same_affine_matrix': (1, 2, 3)
#     },
# }

# same = set()
# different = set()

# for transforms in permutations(transform_functions, 2):
#     a = amb(Point(1,1))
#     b = Point(1,1)
#     for t in transforms:
#         a = t(a, *transform_functions[t]['first'])
#         b = t(amb(b), *transform_functions[t]['first']).transform()
#     a = a.transform()

#     if a == b:
#         same.add(tuple(trans.__name__ for trans in transforms))
#     else:
#         different.add(tuple(trans.__name__ for trans in transforms))


# for t in transform_functions:
#     transforms = (t,)*2

#     a = amb(Point(1,1))
#     b = Point(1,1)
#     for t, input_index in zip(transforms, ('first', 'different')):
#         a = t(a, *transform_functions[t][input_index])
#         b = t(amb(b), *transform_functions[t][input_index]).transform()
#     a = a.transform()

#     if a == b:
#         same.add(tuple(trans.__name__ for trans in transforms))
#     else:
#         different.add(tuple(trans.__name__ for trans in transforms))


# print('same')
# pprint(same)
# print('different')
# pprint(different)

