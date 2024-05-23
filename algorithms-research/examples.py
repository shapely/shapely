from shapely.geometry import Point, LineString, Polygon

# Create a Point
point = Point(1.5, 2.5)
print(point)

# Create a LineString
line = LineString([(0, 0), (1, 1), (2, 2)])
print("LineString:", line)

# Create a Polygon
polygon = Polygon([(0, 0), (1, 1), (1, 0)])
print("Polygon:", polygon)
