import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.plotting import plot_polygon

from figures import SIZE, BLUE, GRAY, set_limits

fig = plt.figure(1, figsize=SIZE, dpi=90)

a = Point(1, 1).buffer(1.5)
b = Point(2, 1).buffer(1.5)

# 1
ax = fig.add_subplot(121)

plot_polygon(a, ax=ax, add_points=False, color=GRAY, alpha=0.2)
plot_polygon(b, ax=ax, add_points=False, color=GRAY, alpha=0.2)

c = a.intersection(b)
plot_polygon(c, ax=ax, add_points=False, color=BLUE, alpha=0.5)

ax.set_title('a.intersection(b)')

set_limits(ax, -1, 4, -1, 3)

#2
ax = fig.add_subplot(122)

plot_polygon(a, ax=ax, add_points=False, color=GRAY, alpha=0.2)
plot_polygon(b, ax=ax, add_points=False, color=GRAY, alpha=0.2)

c = a.symmetric_difference(b)
plot_polygon(c, ax=ax, add_points=False, color=BLUE, alpha=0.5)

ax.set_title('a.symmetric_difference(b)')

set_limits(ax, -1, 4, -1, 3)

plt.show()
