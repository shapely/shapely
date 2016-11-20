from matplotlib import pyplot
from shapely.geometry import LineString
from shapely import affinity

from figures import SIZE, BLUE, GRAY, set_limits, plot_line, add_origin

fig = pyplot.figure(1, figsize=SIZE, dpi=90)

line = LineString([(1, 3), (1, 1), (4, 1)])

# 1
ax = fig.add_subplot(121)

plot_line(ax, line, GRAY)
plot_line(ax, affinity.rotate(line, 90, 'center'), BLUE)
add_origin(ax, line, 'center')

ax.set_title(u"90\N{DEGREE SIGN}, default origin (center)")

set_limits(ax, 0, 5, 0, 4)

# 2
ax = fig.add_subplot(122)

plot_line(ax, line, GRAY)
plot_line(ax, affinity.rotate(line, 90, 'centroid'), BLUE)
add_origin(ax, line, 'centroid')

ax.set_title(u"90\N{DEGREE SIGN}, origin='centroid'")

set_limits(ax, 0, 5, 0, 4)

pyplot.show()
