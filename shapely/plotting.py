import numpy as np

import shapely


def _default_ax():
    import matplotlib.pyplot as plt

    ax = plt.gca()
    ax.grid(True)
    ax.set_aspect("equal")
    return ax


def _path_from_polygon(polygon):
    from matplotlib.path import Path

    path = Path.make_compound_path(
        Path(np.asarray(polygon.exterior.coords)[:, :2]),
        *[Path(np.asarray(ring.coords)[:, :2]) for ring in polygon.interiors],
    )
    return path


def plot_polygon(
    polygon,
    ax=None,
    add_points=True,
    color=None,
    facecolor=None,
    edgecolor=None,
    linewidth=None,
    **kwargs
):
    """
    Plot a (Multi)Polygon.

    Parameters
    ----------
    polygon : shapely.Polygon or shapely.MultiPolygon
    ax : matplotlib Axes, default None
        The axes on which to draw the plot. If not specified, will get the
        current active axes or create a new figure.
    add_points : bool, default True
        If True, also plot the coordinates (vertices) as points.

    """
    if ax is None:
        ax = _default_ax()

    from matplotlib import colors
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    if color is None:
        color = "C0"
    color = colors.to_rgba(color)

    if facecolor is None:
        facecolor = list(color)
        facecolor[-1] = 0.3
        facecolor = tuple(facecolor)

    if edgecolor is None:
        edgecolor = color

    if isinstance(polygon, shapely.MultiPolygon):
        path = Path.make_compound_path(
            *[_path_from_polygon(poly) for poly in polygon.geoms]
        )
    else:
        path = _path_from_polygon(polygon)
    patch = PathPatch(
        path, facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth, **kwargs
    )
    ax.add_patch(patch)
    ax.autoscale_view()

    if add_points:
        line = plot_points(polygon, ax=ax, color=color)
        return patch, line

    return patch


def plot_line(line, ax=None, add_points=True, color=None, linewidth=2, **kwargs):
    """
    Plot a (Multi)LineString/LinearRing

    Parameters
    ----------
    line : shapely.LineString or shapely.LinearRing
    ax : matplotlib Axes, default None
        The axes on which to draw the plot. If not specified, will get the
        current active axes or create a new figure.
    add_points : bool, default True
        If True, also plot the coordinates (vertices) as points.


    """
    if ax is None:
        ax = _default_ax()

    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    if color is None:
        color = "C0"

    if isinstance(line, shapely.MultiLineString):
        path = Path.make_compound_path(
            *[Path(np.asarray(mline.coords)[:, :2]) for mline in line.geoms]
        )
    else:
        path = Path(np.asarray(line.coords)[:, :2])

    patch = PathPatch(
        path, facecolor="none", edgecolor=color, linewidth=linewidth, **kwargs
    )
    ax.add_patch(patch)
    ax.autoscale_view()

    if add_points:
        line = plot_points(line, ax=ax, color=color)
        return patch, line

    return patch


def plot_points(geom, ax=None, color=None, marker="o", **kwargs):
    """
    Plot a shapely.Point/MultiPoint or the vertices of any other geometry type.

    Parameters
    ----------
    line : shapely.Geometry
    ax : matplotlib Axes, default None
        The axes on which to draw the plot. If not specified, will get the
        current active axes or create a new figure.

    """
    if ax is None:
        ax = _default_ax()

    coords = shapely.get_coordinates(geom)
    (line,) = ax.plot(
        coords[:, 0], coords[:, 1], linestyle="", marker=marker, color=color, **kwargs
    )
    return line
