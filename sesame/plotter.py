import numpy as np
import warnings
from . import utils

try:
    import matplotlib.pyplot as plt
    mpl_enabled = True
    try:
        from mpl_toolkits import mplot3d
        has3d = True
    except ImportError:
        warnings.warn("3D plotting not available.", RuntimeWarning)
        has3d = False
except ImportError:
    warnings.warn("matplotlib is not available", RuntimeWarning)
    mpl_enabled = False

def plot_line_defects(sys, scale=1e-6, ls='-o'):
    """
    Plot the sites containing additional charges located on lines in 2D. The
    length scale of the graph is 1 micrometer by default.

    Parameters
    ----------
    sys: Builder
        The discretized system.
    scale: float
        Relevant scaling to apply to the axes.
    ls: string
        Line style of the plotted paths.
    """
    if not mpl_enabled:
        raise RuntimeError("matplotlib was not found, but is required "
                           "for plotting.")

    for c in sys.extra_charge_locations:
        xa, ya = c[0]
        xb, yb = c[1]

        _, _, xcoord, ycoord, _ = utils.Bresenham(sys, (xa, ya,0), (xb,yb,0))

        # plot the path of added charges
        plt.plot(sys.xpts[xcoord]/scale, sys.ypts[ycoord]/scale, ls)

    plt.xlabel('x')
    plt.ylabel('y')

    plt.xlim(xmin=0, xmax=sys.xpts[-1]/scale)
    plt.ylim(ymin=0, ymax=sys.ypts[-1]/scale)
    plt.show()

def plot_plane_defects(sys, scale=1e-6):
    """
    Plot the sites containing additional charges located on planes in 3D. The
    length scale of the graph is 1 micrometer by default.

    Parameters
    ----------
    sys: Builder
        The discretized system.
    scale: float
        Relevant scaling to apply to the axes.
    """
    if not mpl_enabled:
        raise RuntimeError("matplotlib was not found, but is required "
                           "for plotting")

    for c in sys.extra_charge_locations:

        _, X, Y, Z = utils.plane_defects_sites(sys, c) 

        X = X / scale
        Y = Y / scale
        Z = Z / scale

        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(1,1,1, projection='3d')
        ax.plot_surface(X, Y, Z)

    ax.mouse_init(rotate_btn=1, zoom_btn=3)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(xmin=0, xmax=sys.xpts[-1]/scale)
    plt.ylim(ymin=0, ymax=sys.ypts[-1]/scale)
    plt.show()

def plot(sys, data, scale=1e-6, cmap='gnuplot', alpha=1):
    """
    Plot a 2D map of a parameter (like mobility) across the system.

    Parameters
    ----------

    sys: Builder
        The discretized system.
    data: numpy array
        One-dimensional array of data with size equal to the size of the system.
    scale: float
        Relevant scaling to apply to the axes.
    cmap: string
        Name of the colormap used by Matplolib.
    alpha: float
        Transparency of the colormap.
    """

    if not mpl_enabled:
        raise RuntimeError("matplotlib was not found, but is required "
                           "for plotsys()")

    xpts, ypts = sys.xpts / scale, sys.ypts / scale
    data = data.reshape(sys.ny, sys.nx)
    plt.pcolor(xpts, ypts, data)
    plt.xlim(xmin=0, xmax=xpts[-1])
    plt.ylim(ymin=0, ymax=ypts[-1])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


