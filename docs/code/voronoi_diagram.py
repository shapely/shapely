from shapely.geometry import MultiPoint
from shapely.ops import voronoi_diagram

from matplotlib import pyplot
from descartes.patch import PolygonPatch
from figures import SIZE, BLUE, GRAY, set_limits

points = MultiPoint([(0, 0), (1, 1), (0, 2), (2, 2), (3, 1), (1, 0)])
regions = voronoi_diagram(points)

fig = pyplot.figure(1, figsize=SIZE, dpi=90)
fig.set_frameon(True)
ax = fig.add_subplot(111)

for region in regions:
    patch = PolygonPatch(region, facecolor=BLUE, edgecolor=BLUE, alpha=0.5, zorder=2)
    ax.add_patch(patch)

for point in points:
    pyplot.plot(point.x, point.y, 'o', color=GRAY)

set_limits(ax, -1, 4, -1, 3)

pyplot.show()
