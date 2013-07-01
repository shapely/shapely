from itertools import islice, izip

def signed_area(ring):
    """Return the signed area enclosed by a ring in linear time using the 
    algorithm at: http://www.cgafaq.info/wiki/Polygon_Area.
    """
    xs, ys = ring.coords.xy
    xs.append(xs[1])
    ys.append(ys[1])
    return sum(xs[i]*(ys[i+1]-ys[i-1]) for i in range(1, len(ring.coords)))/2.0

def is_ccw_impl(name):
    """Predicate implementation"""
    def is_ccw_op(ring):
        return signed_area(ring) >= 0.0
    return is_ccw_op

