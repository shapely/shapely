from math import sin, cos, tan, pi

def affine(geom, matrix):
    """Return a transformed geometry using an affine transformation matrix

    The coefficient matrix is provided as a list or tuple with 6 or 12 items
    for 2D or 3D transformations, respectively.

    For 2D affine transformations, the 6 parameter matrix is:
        [a, b, d, e, xoff, yoff]
    which represents the transformation matrix:
        / a  b  xoff \
        | d  e  yoff |
        \ 0  0    1  /
    where the vertices are transformed as follows:
        x' = a*x + b*y + xoff
        y' = d*x + e*y + yoff

    For 3D affine transformations, the 12 parameter matrix is:
        [a, b, c, d, e, f, g, h, i, xoff, yoff, zoff]
    which represents the transformation matrix:
        / a  b  c  xoff \
        | d  e  f  yoff |
        | g  h  i  zoff |
        \ 0  0  0    1  /
    where the vertices are transformed as follows:
        x' = a*x + b*y + c*z + xoff
        y' = d*x + e*y + f*z + yoff
        z' = g*x + h*y + i*z + zoff
    """
    if len(matrix) == 6:
        ndim = 2
        a, b, d, e, xoff, yoff = matrix
        if geom.has_z:
            c = f = g = h = i = 1.0
            zoff = 0.0
    elif len(matrix) == 12:
        ndim = 3
        a, b, c, d, e, f, g, h, i, xoff, yoff, zoff = matrix
        if not geom.has_z:
            ndim = 2
            matrix = a, b, d, e, xoff, yoff
    else:
        raise ValueError("'matrix' expects either 6 or 12 coefficients")
    def affine_pts(pts): # internal function
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
    if geom.type in ('Point', 'LineString'):
        return type(geom)(list(affine_pts(geom.coords)))
    elif geom.type == 'Polygon':
        ring = geom.exterior
        shell = type(ring)(list(affine_pts(ring.coords)))
        holes = list(geom.interiors)
        for pos, ring in enumerate(holes):
            holes[pos] = type(ring)(list(affine_pts(ring.coords)))
        return type(geom)(shell, holes)
    elif geom.type.startswith('Multi') or geom.type == 'GeometryCollection':
        return type(geom)([affine(part, matrix) for part in geom.geoms])
    else:
        raise ValueError('Type %r not recognized'%geom.type)

def rotate(geom, angle, origin=None, use_degrees=True):
    """Return a rotated geometry on a 2D plane

    The angle of rotation can be specified in either degrees (default) or
    radians by setting `use_degrees=False`. Positive rotations are clockwise.

    The point of origin can be specified as a Point, otherwise the default
    origin is the centroid of the geometry.

    The transformation matrix for 2D rotation is:
        / cos(r) -sin(r)  xoff \
        | sin(r)  cos(r)  yoff |
        \   0       0       1  /
    where the offsets are calculated from the origin Point(x0, y0):
        xoff = x0 - cos(r) * x0 + sin(r) * y0
        yoff = y0 - sin(r) * x0 - cos(r) * y0
    """
    if use_degrees:
        angle *= pi/180.0
    cosp = cos(angle)
    sinp = sin(angle)
    if abs(cosp) < 2e-16:
        cosp = 0.0
    if abs(sinp) < 2e-16:
        sinp = 0.0
    if origin is None:
        origin = geom.centroid
    elif not (hasattr(origin, 'type') and origin.type == 'Point'):
        raise TypeError("'origin' must be either 'None' or a 'Point' of origin")
    x0, y0 = origin.coords[0][0:2]
    matrix = (cosp, -sinp, 0,
              sinp,  cosp, 0,
              0, 0, 1,
              x0 - cosp * x0 + sinp * y0, y0 - sinp * x0 - cosp * y0, 0)
    return affine(geom, matrix)

def scale(geom, xfact=1.0, yfact=1.0, zfact=1.0, origin=None):
    """Return a scaled geometry, scaled by factors in x, y and/or z directions

    The point of origin can be specified as a Point, otherwise the default
    origin is the centroid of the geometry.

    The general 3D affine transformation matrix for scaling is:
        / xfact  0    0    xoff \
        |   0  yfact  0    yoff |
        |   0    0  zfact  zoff |
        \   0    0    0     1   /
    where the offsets are calculated from the origin Point(x0, y0, z0):
        xoff = x0 - x0 * xfact
        yoff = y0 - y0 * yfact
        zoff = z0 - z0 * zfact
    """
    if origin is None:
        origin = geom.centroid
    elif not (hasattr(origin, 'type') and origin.type == 'Point'):
        raise TypeError("'origin' must be either 'None' or a 'Point' of origin")
    if geom.has_z and origin.has_z:
        x0, y0, z0 = origin.coords[0]
    else:
        x0, y0 = origin.coords[0][0:2]
        z0 = 0.0
    matrix = (xfact, 0.0, 0.0,
              0.0, yfact, 0.0,
              0.0, 0.0, zfact,
              x0 - x0 * xfact, y0 - y0 * yfact, z0 - z0 * zfact)
    return affine(geom, matrix)

def skew(geom, xshear=0.0, yshear=0.0, origin=None, use_degrees=True):
    """Return a skewed geometry, sheared by angles in x, and/or y directions

    The shear angle can be specified in either degrees (default) or radians by
    setting `use_degrees=False`.

    The point of origin can be specified as a Point, otherwise the default
    origin is the centroid of the geometry.

    The general 2D affine transformation matrix for skewing is:
        /   1    tan(xs) 0  xoff \
        | tan(ys)  1     0  yoff |
        \   0      0     0   1   /
    where the offsets are calculated from the origin Point(x0, y0):
        xoff = -x0 * tan(xs)
        yoff = -y0 * tan(ys)
    """
    if use_degrees:
        xshear *= pi/180.0
        yshear *= pi/180.0
    tanx = tan(xshear)
    tany = tan(yshear)
    if abs(tanx) < 2e-16:
        tanx = 0.0
    if abs(tany) < 2e-16:
        tany = 0.0
    if origin is None:
        origin = geom.centroid
    elif not (hasattr(origin, 'type') and origin.type == 'Point'):
        raise TypeError("'origin' must be either 'None' or a 'Point' of origin")
    x0, y0 = origin.coords[0][0:2]
    matrix = (1.0, tanx, 0.0,
              tany, 1.0, 0.0,
              0.0, 0.0, 1.0,
              -x0 * tanx, -y0 * tany, 0.0)
    return affine(geom, matrix)

def translate(geom, x=0.0, y=0.0, z=0.0):
    """Return a translated geometry shifted by offsets along x, y and/or z"""
    matrix = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, x, y, z)
    return affine(geom, matrix)
