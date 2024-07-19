import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import Polygon

def plotting(polygon: Polygon, partition_result):
    """
    Plot the given polygon and its partition.

    Args:
        polygon (Polygon): The polygon to plot and partition.
    """

    # Create a figure and an axes
    fig, ax = plt.subplots()

    # Create a Polygon patch and add it to the plot
    polygon_patch = MplPolygon(
        list(polygon.exterior.coords),
        closed=True,
        edgecolor="blue",
        facecolor="lightblue",
    )
    ax.add_patch(polygon_patch)

    # Plot the LineString objects in a different color
    for line in partition_result:
        x, y = line.xy
        ax.plot(x, y, color="red")

    # Calculate the bounds of the polygon
    min_x, min_y, max_x, max_y = polygon.bounds

    # Add a margin to the bounds to ensure the shape is not cut off
    margin = 1  # Adjust the margin as needed
    ax.set_xlim(min_x - margin, max_x + margin)
    ax.set_ylim(min_y - margin, max_y + margin)

    # Set the aspect of the plot to be equal
    ax.set_aspect("equal")

    # Show the plot
    plt.show()
