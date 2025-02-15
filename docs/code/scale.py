import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely import affinity
from shapely.plotting import plot_polygon

from figures import SIZE, BLUE, GRAY, set_limits, add_origin

fig = plt.figure(1, figsize=SIZE, dpi=90)

triangle = Polygon([(1, 1), (2, 3), (3, 1)])

# 1
ax = fig.add_subplot(121)

plot_polygon(triangle, ax=ax, add_points=False, color=GRAY, alpha=0.5)
triangle_a = affinity.scale(triangle, xfact=1.5, yfact=-1)
plot_polygon(triangle_a, ax=ax, add_points=False, color=BLUE, alpha=0.5)

add_origin(ax, triangle, 'center')

ax.set_title("a) xfact=1.5, yfact=-1")

set_limits(ax, 0, 5, 0, 4)

# 2
ax = fig.add_subplot(122)

plot_polygon(triangle, ax=ax, add_points=False, color=GRAY, alpha=0.5)
triangle_b = affinity.scale(triangle, xfact=2, origin=(1, 1))
plot_polygon(triangle_b, ax=ax, add_points=False, color=BLUE, alpha=0.5)

add_origin(ax, triangle, (1, 1))

ax.set_title("b) xfact=2, origin=(1, 1)")

set_limits(ax, 0, 5, 0, 4)

plt.show()
