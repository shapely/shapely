import matplotlib.pyplot as plt
from shapely.geometry import MultiPoint
from shapely.ops import triangulate
from shapely.plotting import plot_polygon, plot_points

from figures import SIZE, BLUE, GRAY, set_limits

points = MultiPoint([(0, 0), (1, 1), (0, 2), (2, 2), (3, 1), (1, 0)])
triangles = triangulate(points)

fig = plt.figure(1, figsize=SIZE, dpi=90)

ax = fig.add_subplot(111)

for triangle in triangles:
    plot_polygon(triangle, ax=ax, add_points=False, color=BLUE)

plot_points(points, ax=ax, color=GRAY)

set_limits(ax, -1, 4, -1, 3)

plt.show()
