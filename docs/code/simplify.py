import matplotlib.pyplot as plt
from shapely.geometry import MultiPoint, Point
from shapely.plotting import plot_polygon

from figures import SIZE, BLUE, GRAY, set_limits

fig = plt.figure(1, figsize=SIZE, dpi=90)

p = Point(1, 1).buffer(1.5)

# 1
ax = fig.add_subplot(121)

q = p.simplify(0.2)

plot_polygon(p, ax=ax, add_points=False, color=GRAY, alpha=0.5)
plot_polygon(q, ax=ax, add_points=False, color=BLUE, alpha=0.5)

ax.set_title('a) tolerance 0.2')

set_limits(ax, -1, 3, -1, 3)

#2
ax = fig.add_subplot(122)

r = p.simplify(0.5)

plot_polygon(p, ax=ax, add_points=False, color=GRAY, alpha=0.5)
plot_polygon(r, ax=ax, add_points=False, color=BLUE, alpha=0.5)

ax.set_title('b) tolerance 0.5')

set_limits(ax, -1, 3, -1, 3)

plt.show()
