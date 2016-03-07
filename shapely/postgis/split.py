from shapely.geometry import Point, MultiPoint, LineString, MultiLineString, Polygon, MultiPolygon, GeometryCollection
from shapely.ops import polygonize

def _split_polygon_with_line(poly, splitter):
	"""Split a Polygon with a LineString"""

	assert(isinstance(poly, Polygon))
	assert(isinstance(splitter, LineString))

	union = poly.boundary.union(splitter)

	# some polygonized geometries may be holes, we do not want them
	# that's why we test if the original polygon (poly) contains 
	# an inner point of polygonized geometry (pg)
	return [pg for pg in polygonize(union) if poly.contains(pg.representative_point())]

def _split_line_with_line(line, splitter):
	"""Split a LineString with another (Multi)LineString or (Multi)Polygon"""
	
	# if splitter is a polygon, pick it's boundary
	if splitter.type in ('Polygon', 'MultiPolygon'):
		splitter = splitter.boundary

	assert(isinstance(line, LineString))
	assert(isinstance(splitter, LineString) or isinstance(splitter, MultiLineString))
	
	if splitter.crosses(line):
		# The lines cross --> return multilinestring from the split
		return line.difference(splitter)
	elif splitter.relate_pattern(line, '1********'):
		# The lines overlap at some segment (linear intersection of interiors)
		raise ValueError('Input geometry segment overlaps with the splitter.')
	else:
		# The lines do not cross --> return collection with identity line
		return [line]
		
def _split_line_with_point(line, splitter):
	"""Split a LineString with a Point"""

	assert(isinstance(line, LineString))
	assert(isinstance(splitter, Point))

	# check if point is in the interior of the line
	if not line.relate_pattern(splitter, '0********'):
		# point not on line interior --> return collection with single identity line
		# (REASONING: Returning a list with the input line reference and creating a 
		# GeometryCollection at the general split function prevents unnecessary copying 
		# of linestrings in multipoint splitting function)
		return [line]

	# point is on line, get the distance from the first point on line
	distance_on_line = line.project(splitter)
	coords = list(line.coords)
	# split the line at the point and create two new lines
	# TODO: can optimize this by accumulating the computed point-to-point distances
	for i, p in enumerate(coords):
		pd = line.project(Point(p))
		if pd == distance_on_line:
			return [
				LineString(coords[:i+1]), 
				LineString(coords[i:])
			]
		elif distance_on_line < pd:
			# we must interpolate here because the line might use 3D points
			cp = line.interpolate(distance_on_line)
			ls1_coords = coords[:i]
			ls1_coords.append(cp.coords[0])
			ls2_coords = [cp.coords[0]]
			ls2_coords.extend(coords[i:])
			return [LineString(ls1_coords), LineString(ls2_coords)]

def _split_line_with_multipoint(line, splitter):
	"""Split a LineString with a MultiPoint"""

	assert(isinstance(line, LineString))
	assert(isinstance(splitter, MultiPoint))
	
	chunks = [line]
	for pt in splitter.geoms:
		new_chunks = []
		for chunk in chunks:
			# add the newly split 2 lines or the same line if not split
			new_chunks.extend(split_line_with_point(chunk, pt))
		chunks = new_chunks
	
	return chunks


def split_impl(geom, splitter):
	"""Split a geometry with another geometry"""

	if geom.type in ('MultiLineString', 'MultiPolygon'):
		 return GeometryCollection([i for part in geom.geoms for i in split(part, splitter).geoms])

	elif geom.type == 'LineString':
		if splitter.type in ('LineString', 'MultiLineString', 'Polygon', 'MultiPolygon'):
			split_func = split_line_with_line
		elif splitter.type in ('Point'):
			split_func = split_line_with_point
		elif splitter.type in ('MultiPoint'):
			split_func =  split_line_with_multipoint
		else:
			raise ValueError("Splitting a LineString with a %s is not supported" % splitter.type)

	elif geom.type == 'Polygon':
		if splitter.type == 'LineString':
			split_func = split_polygon_with_line
		else:
			raise ValueError("Splitting a Polygon with a %s is not supported" % splitter.type)

	else:
		raise ValueError("Splitting %s geometry is not supported" % geom.type)

	return GeometryCollection(split_func(geom, splitter))
