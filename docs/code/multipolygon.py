import matplotlib.pyplot as plt
from shapely.geometry import MultiPolygon
from shapely.plotting import plot_polygon

from figures import SIZE, BLUE, RED, set_limits
    
fig = plt.figure(1, figsize=SIZE, dpi=90)

# 1: valid multi-polygon
ax = fig.add_subplot(121)

a = [(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)]
b = [(1, 1), (1, 2), (2, 2), (2, 1), (1, 1)]

multi1 = MultiPolygon([[a, []], [b, []]])

for polygon in multi1.geoms:
    plot_polygon(polygon, ax=ax, color=BLUE)

ax.set_title('a) valid')

set_limits(ax, -1, 3, -1, 3)

#2: invalid self-touching ring
ax = fig.add_subplot(122)

c = [(0, 0), (0, 1.5), (1, 1.5), (1, 0), (0, 0)]
d = [(1, 0.5), (1, 2), (2, 2), (2, 0.5), (1, 0.5)]

multi2 = MultiPolygon([[c, []], [d, []]])

for polygon in multi2.geoms:
    plot_polygon(polygon, ax=ax, color=RED)

ax.set_title('b) invalid')

set_limits(ax, -1, 3, -1, 3)

plt.show()
