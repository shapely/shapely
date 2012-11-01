def affine(geom, matrix):
    """Perform an affine transformation on a geometry

    The coefficient matrix is provided as a list or tuple with 6 or 12 items
    for 2D or 3D transformations, respectively.

    For 2D affine transformations, the 6 parameter matrix is:
        [a, b, d, e, xoff, yoff]
    which represents the transformation matrix:
        / a  b  xoff \
        | d  e  yoff |
        \ 0  0     1 /
    where the vertices are transformed as follows:
        x' = a*x + b*y + xoff
        y' = d*x + e*y + yoff

    For 3D affine transformations, the 12 parameter matrix is:
        [a, b, c, d, e, f, g, h, i, xoff, yoff, zoff]
    which represents the transformation matrix:
        / a  b  c xoff \
        | d  e  f yoff |
        | g  h  i zoff |
        \ 0  0  0    1 /
    where the vertices are transformed as follows:
        x' = a*x + b*y + c*z + xoff
        y' = d*x + e*y + f*z + yoff
        z' = g*x + h*y + i*z + zoff
    """
    matrix = [float(x) for x in matrix]
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

