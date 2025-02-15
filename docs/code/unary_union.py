import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.ops import unary_union
from shapely.plotting import plot_polygon

from figures import SIZE, BLUE, GRAY, set_limits

polygons = [Point(i, 0).buffer(0.7) for i in range(5)]

fig = plt.figure(1, figsize=SIZE, dpi=90)

# 1
ax = fig.add_subplot(121)

for ob in polygons:
    plot_polygon(ob, ax=ax, add_points=False, color=GRAY)

ax.set_title('a) polygons')

set_limits(ax, -2, 6, -2, 2)

#2
ax = fig.add_subplot(122)

u = unary_union(polygons)
plot_polygon(u, ax=ax, add_points=False, color=BLUE)

ax.set_title('b) union')

set_limits(ax, -2, 6, -2, 2)

plt.show()
