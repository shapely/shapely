"""Affine transforms, both in general and specific, named transforms."""
from collections import namedtuple
import numpy as np
from itertools import chain
from more_itertools import first
import numba

sg = None  # shapely.geometry will be put here

__all__ = ['affine_matrix_builder', 'affine_transform', 'rotate', 'scale', 'skew', 'translate']


class affine_matrix_builder:
    """
    Class for building affine transformation matrices.
    Reduces the number of transformations needed by combining the
    affine matrices when appropriate.
    Class implemented by William Rusnack, github.com/BebeSparkelSparkel, williamrusnack@gmail.com
    """
    # import shapely.geometry as sg

    def __init__(self, geom):
        global sg

        self.geom = geom

        self.transforms_to_apply = list()

        try:
            sg.base
        except AttributeError:
            import shapely.geometry
            sg = shapely.geometry

    _transform_log = namedtuple('_transform_log',
        ('transform', 'inputs', 'matrix', 'offsets'))

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
        if not isinstance(matrix, (tuple, list)):
            raise TypeError('Matrix must be a tuple or list.')
        if len(matrix) not in (6, 12):
            raise ValueError('Wrong size matrix.')

        self.transforms_to_apply.append(self._transform_log(
                affine_matrix_builder.affine_transform,
                dict(matrix=matrix),
                *_to_np_matrix(matrix)
            ))

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
        if not use_radians:  # convert from degrees
            angle *= np.pi / 180.0

        cosp = np.cos(angle)
        if abs(cosp) < 2.5e-16: cosp = 0.0
        sinp = np.sin(angle)
        if abs(sinp) < 2.5e-16: sinp = 0.0

        x0, y0, extra = self._interpret_origin(origin)

        self.transforms_to_apply.append(self._transform_log(
                transform = affine_matrix_builder.rotate,
                inputs = dict(angle=angle, origin=origin, use_radians=use_radians),
                matrix = np.mat(((cosp, -sinp, 0.0),
                                    (sinp,  cosp, 0.0),
                                    (0.0,    0.0, 1.0)), dtype=np.float),
                offsets = np.array((
                    x0 - x0 * cosp + y0 * sinp,
                    y0 - x0 * sinp - y0 * cosp,
                    0.0), dtype=np.float),
            ))

        return self

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
        x0, y0, z0 = self._interpret_origin(origin)

        self.transforms_to_apply.append(self._transform_log(
                transform = affine_matrix_builder.scale,
                inputs = dict(xfact=xfact, yfact=yfact, zfact=zfact, origin=origin),
                matrix = np.mat(((xfact, 0.0, 0.0),
                                    (0.0, yfact, 0.0),
                                    (0.0, 0.0, zfact)),
                                dtype=np.float),
                offsets = np.array((
                    x0 - x0 * xfact,
                    y0 - y0 * yfact,
                    z0 - z0 * zfact),
                    dtype=np.float),
            ))

        return self

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
        if not use_radians:  # convert from degrees
            xs *= np.pi / 180.0
            ys *= np.pi / 180.0

        tanx = np.tan(xs)
        if abs(tanx) < 2.5e-16: tanx = 0.0
        tany = np.tan(ys)
        if abs(tany) < 2.5e-16: tany = 0.0

        x0, y0, extra = self._interpret_origin(origin)

        self.transforms_to_apply.append(self._transform_log(
                transform = affine_matrix_builder.skew,
                inputs = dict(xs=xs, ys=ys, origin=origin, use_radians=use_radians),
                matrix = np.mat(((1.0, tanx, 0.0),
                                    (tany, 1.0, 0.0),
                                    (0.0,  0.0, 1.0)), dtype=np.float),
                offsets = np.array((
                    -y0 * tanx,
                    -x0 * tany,
                    0.0), dtype=np.float),
            ))

        return self

    def translate(self, xoff=0.0, yoff=0.0, zoff=0.0):
        """
        Applies a translate matrix that shifts by offsets along each dimension.

        The general 3D affine transformation matrix for translation is:

            / 1  0  0 xoff \
            | 0  1  0 yoff |
            | 0  0  1 zoff |
            \ 0  0  0   1  /
        """
        # matrix = (1.0, 0.0, 0.0,
        #           0.0, 1.0, 0.0,
        #           0.0, 0.0, 1.0,
        #           xoff, yoff, zoff)

        self.transforms_to_apply.append(self._transform_log(
                transform = affine_matrix_builder.translate,
                inputs = dict(xoff=xoff, yoff=yoff, zoff=zoff),
                matrix = np.mat(np.identity(3, dtype=np.float)),
                offsets = np.array((xoff, yoff, zoff), dtype=np.float),
            ))

        return self

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
        if not self.transforms_to_apply:
            raise TypeError('No transforms to apply.')

        if self.geom.is_empty:
            return self.geom
        # import pdb; pdb.set_trace();
        to_apply = iter(self.transforms_to_apply)
        current_trans = next(to_apply)
        matrix = np.matrix(current_trans.matrix)
        offsets = np.array(current_trans.offsets)
        geom = self.geom
        for next_trans in to_apply:
            if (current_trans.transform, next_trans.transform) in self._compatable_combinations:
                matrix *= np.mat(next_trans.matrix)
                offsets += next_trans.offsets
            else:
                geom = affine_transform(geom,
                    np.concatenate((matrix.A1, offsets)))
                matrix = np.matrix(next_trans.matrix)
                offsets = np.array(next_trans.offsets)

            current_trans = next_trans

        return affine_transform(geom,
            np.concatenate((matrix.A1, offsets)))


    # def _combine_matrices(self):
    #     """
    #     Returns true if the affine if the last and current matrices can
    #     be safely combined to get the expected output.

    #     IMPROVEMENT
    #     this can be optimized more to look at inputs/matrix too but has not been done yet
    #     it seems like transfromations with the same origin can be combined but that should be proven true
    #     inputs are also saved in self.last_transform and self.current_transform which will allow origin comparisons
    #     I am also not sure if the total history matters in combining affine matrices or if only the last affine transform matters
    #     """
    #     return (self.last_transform.transform, self.current_transform.transform) \
    #         in self._compatable_combinations

    def _interpret_origin(self, origin):
        """Returns interpreted coordinate tuple for origin parameter.

        This is a helper function for other transform functions.

        The point of origin can be a keyword 'center' for the 2D bounding box
        center, 'centroid' for the geometry's 2D centroid, a Point object or a
        coordinate tuple (x0, y0, z0).
        """

        # get coordinate tuple from 'origin' from keyword or Point type
        if isinstance(origin, str):
            if origin == 'center':
                # bounding box center
                minx, miny, maxx, maxy = self.geom.bounds
                origin = np.array(((maxx + minx) / 2.0, (maxy + miny) / 2.0, 0.0))

            elif origin == 'centroid':
                origin = self.geom.centroid.coords[0]

            else:
                raise ValueError("'origin' keyword %r is not recognized" % origin)

        elif isinstance(origin, sg.Point):
            origin = origin.coords[0]

        if len(origin) == 3:
            origin = np.array(origin)
        elif len(origin) == 2:
            origin = np.array(tuple(chain(origin, (0.0,))))
        else:
            raise ValueError("Expected number of items in 'origin' to be either 2 or 3")

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


# def _apply_matrix(geom, matrix, offsets):
#     if not geom.has_z:
#         matrix = np.mat(matrix[:2,:2])
#         offsets = offsets[:2]

#     # Process coordinates from each supported geometry type
#     if isinstance(geom, sg.Point):
#         return type(geom)(first(affine_pts(np.array(geom.coords), matrix, offsets)))
#     if isinstance(geom, (sg.Point, sg.LineString, sg.LinearRing)):
#         return type(geom)(tuple(affine_pts(np.array(geom.coords), matrix, offsets)))

#     elif isinstance(geom, sg.Polygon):
#         return sg.Polygon(
#                 sg.LinearRing(tuple(
#                         affine_pts(np.array(geom.exterior.coords), matrix, offsets)
#                     )),
#                 tuple(
#                         sg.LinearRing(tuple(affine_pts(np.array(ring.coords), matrix, offsets)))
#                         for ring in geom.interiors
#                     ),
#             )

#     elif isinstance(geom, sg.base.BaseMultipartGeometry):
#         # Recursive call
#         # TODO: fix GeometryCollection constructor
#         return type(geom)(tuple(
#                 affine_transform(part, self.matrix)
#                 for part in geom.geoms
#             ))

#     else:
#         raise ValueError('Type {} not recognized'.format(geom.type))

# @numba.jit(nopython=True)
# def affine_pts(pts, matrix, offsets):
#     """Internal function to yield affine transform of coordinate tuples"""
#     for pt in pts:
#         yield (matrix * np.mat(pt, dtype=float).transpose()).A1 + offsets


def _to_np_matrix(matrix):
    '''
    Converts the matrix to a numpy 3x3 matrix and an array of offsets.
    If matrix is None returns identity matrix and offsets are zero.
    '''
    if matrix is None:
        matrix_out = np.mat(np.identity(3, dtype=np.float))
        offsets = np.zeros(3, dtype=np.float)
        return matrix_out, offsets

    matrix = tuple(matrix)

    if len(matrix) == 6:
        # creating a 3x3 matrix. Undefined values are set to zero
        matrix_out = np.mat([
                matrix[:2] + (0,),
                matrix[2:4] + (0,),
                (0, 0, 1)
            ], dtype=np.float)
        offsets = np.array(matrix[4:] + (0,), dtype=np.float)

    elif len(matrix) == 12:
        matrix_out = np.array(matrix[:9], dtype=np.float).reshape(3,3)
        offsets = np.array(matrix[9:], dtype=np.float)

    else:
        raise ValueError('Affine transform matrix must have len of 6 or 9 but has len of ' + str(len(matrix)))

    return matrix_out, offsets


# def square_matrix_size(matrix):
#     size = len(matrix)**.5
#     if size != int(size) or size < 1:
#         raise ValueError('Not a square matrix')
#     return int(size)

# def row_iter(matrix):
#     square_size = square_matrix_size(matrix)

#     for i in range(0, len(matrix), square_size):
#         yield matrix[i:i + square_size]

# def get_column(matrix, column_index):
#     return (row[column_index] for row in row_iter(matrix))

# def column_iter(matrix):
#     square_size = square_matrix_size(matrix)

#     for i in range(square_size):
#         yield get_column(matrix, i)

# def multiply_matrices(a, b):
#     result = []
#     for row in row_iter(a):
#         for col in column_iter(b):
#             result.append(sum(r*c for r, c in zip(row, col)))
#     return tuple(result)



# for backwards compatability
def rotate(geom, angle, origin='center', use_radians=False):
    return affine_matrix_builder(geom).rotate(angle, origin, use_radians).transform()# def rotate(geom, angle, origin='center', use_radians=False):

def scale(geom, xfact=1.0, yfact=1.0, zfact=1.0, origin='center'):
    return affine_matrix_builder(geom).scale(xfact, yfact, zfact, origin).transform()

def skew(geom, xs=0.0, ys=0.0, origin='center', use_radians=False):
    return affine_matrix_builder(geom).skew(xs, ys, origin, use_radians).transform()

def translate(geom, xoff=0.0, yoff=0.0, zoff=0.0):
    return affine_matrix_builder(geom).translate(xoff, yoff, zoff).transform()

# def affine_transform(geom, matrix):
#     return affine_matrix_builder(geom, matrix).transform()
def affine_transform(geom, matrix):
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
    if geom.is_empty:
        return geom

    if len(matrix) == 6:
        ndim = 2
        a, b, d, e, xoff, yoff = matrix
        if geom.has_z:
            ndim = 3
            i = 1.0
            c = f = g = h = zoff = 0.0
            matrix = a, b, c, d, e, f, g, h, i, xoff, yoff, zoff
    elif len(matrix) == 12:
        ndim = 3
        a, b, c, d, e, f, g, h, i, xoff, yoff, zoff = matrix
        if not geom.has_z:
            ndim = 2
            matrix = a, b, d, e, xoff, yoff
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

    # # Process coordinates from each supported geometry type
    # if geom.type in ('Point', 'LineString', 'LinearRing'):
    #     return type(geom)(list(affine_pts(geom.coords)))
    # elif geom.type == 'Polygon':
    #     ring = geom.exterior
    #     shell = type(ring)(list(affine_pts(ring.coords)))
    #     holes = list(geom.interiors)
    #     for pos, ring in enumerate(holes):
    #         holes[pos] = type(ring)(list(affine_pts(ring.coords)))
    #     return type(geom)(shell, holes)
    # elif geom.type.startswith('Multi') or geom.type == 'GeometryCollection':
    #     # Recursive call
    #     # TODO: fix GeometryCollection constructor
    #     return type(geom)([affine_transform(part, matrix)
    #                        for part in geom.geoms])
    # else:
    #     raise ValueError('Type %r not recognized' % geom.type)
    # Process coordinates from each supported geometry type
    if isinstance(geom, sg.Point):
        return type(geom)(first(affine_pts(np.array(geom.coords), matrix, offsets)))
    if isinstance(geom, (sg.Point, sg.LineString, sg.LinearRing)):
        return type(geom)(tuple(affine_pts(np.array(geom.coords), matrix, offsets)))

    elif isinstance(geom, sg.Polygon):
        return sg.Polygon(
                sg.LinearRing(tuple(
                        affine_pts(np.array(geom.exterior.coords), matrix, offsets)
                    )),
                tuple(
                        sg.LinearRing(tuple(affine_pts(np.array(ring.coords), matrix, offsets)))
                        for ring in geom.interiors
                    ),
            )

    elif isinstance(geom, sg.base.BaseMultipartGeometry):
        # Recursive call
        # TODO: fix GeometryCollection constructor
        return type(geom)(tuple(
                affine_transform(part, self.matrix)
                for part in geom.geoms
            ))

    else:
        raise ValueError('Type {} not recognized'.format(geom.type))


# # code used to determine how affine matrices can be combined
# from itertools import permutations, chain

# from shapely.geometry import Point
# from shapely.affinity import affine_matrix_builder as amb


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


