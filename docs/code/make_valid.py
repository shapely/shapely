from matplotlib import pyplot as plt
import shapely
import shapely.plotting as plotter

input = shapely.MultiPolygon(
    [
        shapely.Polygon(
            [(2, 0), (2, 12), (7, 12), (7, 10), (7, 12), (10, 12), (8, 12), (8, 0), (2, 0)],
            [[(3, 10), (5, 10), (5, 12), (3, 12), (3, 10)]]
        ),
        shapely.Polygon(
            [(4, 2), (4, 8), (12, 8), (12, 2), (4, 2)],
            [[(6, 4), (10, 4), (10, 6), (6, 6), (6, 4)]],
        )
    ]
)

fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(11, 4)) 
plotter.plot_polygon(input, ax=ax[0], linewidth=2)
ax[0].set_title("invalid input") 

# Structure makevalid
valid_structure = shapely.make_valid(input, method="structure", keep_collapsed=True)
plotter.plot_polygon(valid_structure, ax=ax[1], linewidth=2)
ax[1].set_title("structure")

# Linework makevalid
valid_linework = shapely.make_valid(input)
for geom in valid_linework.geoms:
    if isinstance(geom, shapely.MultiPolygon):
        plotter.plot_polygon(geom, ax=ax[2], linewidth=2)
    else:
        plotter.plot_line(geom, ax=ax[2], linewidth=2)
    ax[2].set_title("linework") 
plt.show()
fig.savefig("make_valid.png")
