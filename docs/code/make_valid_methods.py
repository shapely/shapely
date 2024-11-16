from matplotlib import pyplot as plt
import shapely
from shapely.plotting import plot_points, plot_polygon, plot_line

from figures import BLUE, GRAY, RED

input = shapely.MultiPolygon(
    [
        shapely.Polygon(
            [
                (2, 0),
                (2, 12),
                (7, 12),
                (7, 10),
                (7, 12),
                (10, 12),
                (8, 12),
                (8, 0),
                (2, 0),
            ],
            [[(3, 10), (5, 10), (5, 12), (3, 12), (3, 10)]],
        ),
        shapely.Polygon(
            [(4, 2), (4, 8), (12, 8), (12, 2), (4, 2)],
            [[(6, 4), (10, 4), (10, 6), (6, 6), (6, 4)]],
        ),
    ]
)

fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(11, 4), dpi=90)
plot_polygon(input, ax=ax[0], add_points=False, color=BLUE)
plot_points(input, ax=ax[0], color=GRAY, alpha=0.7)
ax[0].set_title("invalid input")
ax[0].set_aspect("equal")

# Structure makevalid
valid_structure = shapely.make_valid(input, method="structure", keep_collapsed=True)
plot_polygon(valid_structure, ax=ax[1], add_points=False, color=BLUE)
plot_points(valid_structure, ax=ax[1], color=GRAY, alpha=0.7)

ax[1].set_title("make_valid - structure")
ax[1].set_aspect("equal")

# Linework makevalid
valid_linework = shapely.make_valid(input)
for geom in valid_linework.geoms:
    if isinstance(geom, shapely.MultiPolygon):
        plot_polygon(geom, ax=ax[2], add_points=False, color=BLUE)
        plot_points(geom, ax=ax[2], color=GRAY, alpha=0.7)
    else:
        plot_line(geom, ax=ax[2], color=RED, linewidth=1)
        plot_points(geom, ax=ax[2], color=GRAY, alpha=0.7)
ax[2].set_title("make_valid - linework")
ax[2].set_aspect("equal")

fig.tight_layout()
plt.show()
