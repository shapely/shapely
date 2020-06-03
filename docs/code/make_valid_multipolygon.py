from shapely.geometry import Polygon
from shapely.validation import make_valid

from matplotlib import pyplot
from descartes.patch import PolygonPatch
from figures import SIZE, BLUE, RED, set_limits

invalid_poly = Polygon([(0, 0), (0, 2), (1, 1), (2, 2), (2, 0), (1, 1), (0, 0)])
valid_poly = make_valid(invalid_poly)

fig = pyplot.figure(1, figsize=SIZE, dpi=90)
fig.set_frameon(True)


invalid_ax = fig.add_subplot(121)

patch = PolygonPatch(invalid_poly, facecolor=BLUE, edgecolor=BLUE, alpha=0.5, zorder=2)
invalid_ax.add_patch(patch)

set_limits(invalid_ax, -1, 3, -1, 3)


valid_ax = fig.add_subplot(122)

patch = PolygonPatch(valid_poly[0], facecolor=BLUE, edgecolor=BLUE, alpha=0.5, zorder=2)
valid_ax.add_patch(patch)

patch = PolygonPatch(valid_poly[1], facecolor=RED, edgecolor=RED, alpha=0.5, zorder=2)
valid_ax.add_patch(patch)

set_limits(valid_ax, -1, 3, -1, 3)

pyplot.show()