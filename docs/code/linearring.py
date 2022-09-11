import matplotlib.pyplot as plt
from shapely.geometry import LinearRing
from shapely.plotting import plot_line, plot_points

from figures import SIZE, BLUE, GRAY, RED, set_limits

fig = plt.figure(1, figsize=SIZE, dpi=90)

# 1: valid ring
ax = fig.add_subplot(121)
ring = LinearRing([(0, 0), (0, 2), (1, 1), (2, 2), (2, 0), (1, 0.8), (0, 0)])

plot_line(ring, ax=ax, add_points=False, color=BLUE, alpha=0.7)
plot_points(ring, ax=ax, color=GRAY, alpha=0.7)

ax.set_title('a) valid')

set_limits(ax, -1, 3, -1, 3)

#2: invalid self-touching ring
ax = fig.add_subplot(122)
ring2 = LinearRing([(0, 0), (0, 2), (1, 1), (2, 2), (2, 0), (1, 1), (0, 0)])

plot_line(ring2, ax=ax, add_points=False, color=RED, alpha=0.7)
plot_points(ring2, ax=ax, color=GRAY, alpha=0.7)

ax.set_title('b) invalid')

set_limits(ax, -1, 3, -1, 3)

plt.show()
