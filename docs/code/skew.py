from matplotlib import pyplot
from shapely.wkt import loads as load_wkt
from shapely import affinity
from descartes.patch import PolygonPatch

from figures import SIZE, BLUE, GRAY, set_limits, add_origin

fig = pyplot.figure(1, figsize=SIZE, dpi=90)

# Geometry from JTS TestBuilder with fixed precision model of 100.0
# Using CreateShape > FontGlyphSanSerif and A = triangle.wkt from scale.py
R = load_wkt('''\
POLYGON((2.218 2.204, 2.273 2.18, 2.328 2.144, 2.435 2.042, 2.541 1.895,
  2.647 1.702, 3 1, 2.626 1, 2.298 1.659, 2.235 1.777, 2.173 1.873,
  2.112 1.948, 2.051 2.001, 1.986 2.038, 1.91 2.064, 1.823 2.08, 1.726 2.085,
  1.347 2.085, 1.347 1, 1 1, 1 3.567, 1.784 3.567, 1.99 3.556, 2.168 3.521,
  2.319 3.464, 2.441 3.383, 2.492 3.334, 2.536 3.279, 2.604 3.152,
  2.644 3.002, 2.658 2.828, 2.651 2.712, 2.63 2.606, 2.594 2.51, 2.545 2.425,
  2.482 2.352, 2.407 2.29, 2.319 2.241, 2.218 2.204),
 (1.347 3.282, 1.347 2.371, 1.784 2.371, 1.902 2.378, 2.004 2.4, 2.091 2.436,
  2.163 2.487, 2.219 2.552, 2.259 2.63, 2.283 2.722, 2.291 2.828, 2.283 2.933,
  2.259 3.025, 2.219 3.103, 2.163 3.167, 2.091 3.217, 2.004 3.253, 1.902 3.275,
  1.784 3.282, 1.347 3.282))''')

# 1
ax = fig.add_subplot(121)

patch1a = PolygonPatch(R, facecolor=GRAY, edgecolor=GRAY,
                       alpha=0.5, zorder=1)
skewR = affinity.skew(R, xs=20, origin=(1, 1))
patch1b = PolygonPatch(skewR, facecolor=BLUE, edgecolor=BLUE,
                       alpha=0.5, zorder=2)
ax.add_patch(patch1a)
ax.add_patch(patch1b)

add_origin(ax, R, (1, 1))

ax.set_title("a) xs=20, origin(1, 1)")

set_limits(ax, 0, 5, 0, 4)

# 2
ax = fig.add_subplot(122)

patch2a = PolygonPatch(R, facecolor=GRAY, edgecolor=GRAY,
                       alpha=0.5, zorder=1)
skewR = affinity.skew(R, ys=30)
patch2b = PolygonPatch(skewR, facecolor=BLUE, edgecolor=BLUE,
                       alpha=0.5, zorder=2)
ax.add_patch(patch2a)
ax.add_patch(patch2b)

add_origin(ax, R, 'center')

ax.set_title("b) ys=30")

set_limits(ax, 0, 5, 0, 4)

pyplot.show()
