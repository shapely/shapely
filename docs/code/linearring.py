from matplotlib import pyplot
from shapely.geometry.polygon import LinearRing

from figures import SIZE, set_limits, plot_coords, plot_line_isvalid

fig = pyplot.figure(1, figsize=SIZE, dpi=90)

# 1: valid ring
ax = fig.add_subplot(121)
ring = LinearRing([(0, 0), (0, 2), (1, 1), (2, 2), (2, 0), (1, 0.8), (0, 0)])

plot_coords(ax, ring)
plot_line_isvalid(ax, ring, alpha=0.7)

ax.set_title('a) valid')

set_limits(ax, -1, 3, -1, 3)

#2: invalid self-touching ring
ax = fig.add_subplot(122)
ring2 = LinearRing([(0, 0), (0, 2), (1, 1), (2, 2), (2, 0), (1, 1), (0, 0)])

plot_coords(ax, ring2)
plot_line_isvalid(ax, ring2, alpha=0.7)

ax.set_title('b) invalid')

set_limits(ax, -1, 3, -1, 3)

pyplot.show()

