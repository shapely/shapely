import matplotlib.pyplot as plt
from shapely.geometry import LineString
from shapely import affinity
from shapely.plotting import plot_line

from figures import SIZE, BLUE, GRAY, set_limits, add_origin

fig = plt.figure(1, figsize=SIZE, dpi=90)

line = LineString([(1, 3), (1, 1), (4, 1)])

# 1
ax = fig.add_subplot(121)

plot_line(line, ax=ax, add_points=False, color=GRAY)
plot_line(affinity.rotate(line, 90, 'center'), ax=ax, add_points=False, color=BLUE)
add_origin(ax, line, 'center')

ax.set_title("90\N{DEGREE SIGN}, default origin (center)")

set_limits(ax, 0, 5, 0, 4)

# 2
ax = fig.add_subplot(122)

plot_line(line, ax=ax, add_points=False, color=GRAY)
plot_line(affinity.rotate(line, 90, 'centroid'), ax=ax, add_points=False, color=BLUE)
add_origin(ax, line, 'centroid')

ax.set_title("90\N{DEGREE SIGN}, origin='centroid'")

set_limits(ax, 0, 5, 0, 4)

plt.show()
