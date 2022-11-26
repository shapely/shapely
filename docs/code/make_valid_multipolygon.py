import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.validation import make_valid
from shapely.plotting import plot_polygon

from figures import SIZE, BLUE, RED, set_limits

invalid_poly = Polygon([(0, 0), (0, 2), (1, 1), (2, 2), (2, 0), (1, 1), (0, 0)])
valid_poly = make_valid(invalid_poly)

fig = plt.figure(1, figsize=SIZE, dpi=90)

invalid_ax = fig.add_subplot(121)

plot_polygon(invalid_poly, ax=invalid_ax, add_points=False, color=BLUE)

set_limits(invalid_ax, -1, 3, -1, 3)


valid_ax = fig.add_subplot(122)

plot_polygon(valid_poly.geoms[0], ax=valid_ax, add_points=False, color=BLUE)
plot_polygon(valid_poly.geoms[1], ax=valid_ax, add_points=False, color=RED)

set_limits(valid_ax, -1, 3, -1, 3)

plt.show()
