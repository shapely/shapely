from shapely import geometry


def shift(geom):
    """
    Reads every point in every component of input geometry, and performs the following change:
        if the longitude coordinate is <0, adds 360 to it.
        if the longitude coordinate is >180, subtracts 360 from it.

    Useful for shifting between 0 and 180 centric map
    """

    if geom.is_empty:
        return geom

    if geom.has_z:
        num_dim = 3
    else:
        num_dim = 2

    def shift_pts(pts):
        """Internal function to perform shift of individual points"""
        if num_dim == 2:
            for x, y in pts:
                if x < 0:
                    x += 360
                elif x > 180:
                    x -= 360
                yield (x, y)
        elif num_dim == 3:
            for x, y, z in pts:
                if x < 0:
                    x += 360
                elif x > 180:
                    x -= 360
                yield (x, y, z)

    # Determine the geometry type to call appropriate handler
    if geom.type in ('Point', 'LineString'):
        return type(geom)(list(shift_pts(geom.coords)))
    elif geom.type == 'Polygon':
        ring = geom.exterior
        shell = type(ring)(list(shift_pts(ring.coords)))
        holes = list(geom.interiors)
        for pos, ring in enumerate(holes):
            holes[pos] = type(ring)(list(shift_pts(ring.coords)))
        return type(geom)(shell, holes)
    elif geom.type.startswith('Multi') or geom.type == 'GeometryCollection':
        # Recursive call to shift all components
        return type(geom)([shift(part)
                           for part in geom.geoms])
    else:
        raise ValueError('Type %r not supported' % geom.type)


def idl_resolve(geom, buffer_width=0.0000001):
    """
    Identifies when an intersection is present with -180/180 international date line and corrects it

    Geometry is shifted to 180 centric map and intersection is checked against a line defined as [(180, -90), (180,90)]
    If intersection is identified then the line is buffered by given amount (decimal degrees) and the difference
    between input geometry and buffer result is returned
    If no intersection is identified the passed in geometry is returned
    """

    intersecting_line = geometry.LineString(((180, -90), (180, 90)))

    shifted_geom = shift(geom)

    if shifted_geom.intersects(intersecting_line):
        buffered_line = intersecting_line.buffer(buffer_width)
        difference_geom = shifted_geom.difference(buffered_line)
        geom = shift(difference_geom)

    return geom

