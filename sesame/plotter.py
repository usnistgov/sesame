# Copyright 2017 University of Maryland.
#
# This file is part of Sesame. It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution.

import numpy as np
from . import utils

try:
    import matplotlib.pyplot as plt
    mpl_enabled = True
    try:
        from mpl_toolkits import mplot3d
        has3d = True
    except:
        has3d = False
except:
    mpl_enabled = False

def plot_line_defects(sys, scale=1e-6, ls='-o', show=True):
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
    ax: Maplotlib axis
        A plot is added to it if given. If not given, a new one is created and a
        figure is displayed.
    """
    if not mpl_enabled:
        raise RuntimeError("matplotlib was not found, but is required "
                           "for plotting.")

    if fig is None:
        fig = plt.figure()
        show = True
    # add axis to figure
    ax = fig.add_subplot(111)

    for c in sys.defects_list:
        xa, ya = c.location[0]
        xb, yb = c.location[1]

        _, _, xcoord, ycoord, _ = utils.Bresenham(sys, (xa, ya,0), (xb,yb,0))

        # plot the path of added charges
        ax.plot(sys.xpts[xcoord]/scale, sys.ypts[ycoord]/scale, ls)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
       
    ax.set_xlim(xmin=0, xmax=sys.xpts[-1]/scale)
    ax.set_ylim(ymin=0, ymax=sys.ypts[-1]/scale)

    if show:
        plt.show()

def plot_plane_defects(sys, scale=1e-6, fig=None):
    """
    Plot the sites containing additional charges located on planes in 3D. The
    length scale of the graph is 1 micrometer by default.

    Parameters
    ----------
    sys: Builder
        The discretized system.
    scale: float
        Relevant scaling to apply to the axes.
    fig: Maplotlib figure
        A plot is added to it if given. If not given, a new one is created and 
        displayed.
    """
    if not mpl_enabled:
        raise RuntimeError("matplotlib was not found, but is required "
                           "for plotting")

    show = False
    if fig is None:
        fig = plt.figure()
        show = True
    # add axis to figure
    ax = fig.add_subplot(1,1,1, projection='3d')

    for c in sys.defects_list:

        _, X, Y, Z, _ = utils.plane_defects_sites(sys, c.location) 

        X = X / scale
        Y = Y / scale
        Z = Z / scale

        ax.plot_surface(X, Y, Z)

    ax.mouse_init(rotate_btn=1, zoom_btn=3)

    xLabel = ax.set_xlabel('x')
    yLabel = ax.set_ylabel('y')
    zLabel = ax.set_zlabel('z')

    ax.set_xlim3d(0, sys.xpts[-1]/scale)
    ax.set_ylim3d(0, sys.ypts[-1]/scale)
    ax.set_zlim3d(0, sys.zpts[-1]/scale)

    if show:
        plt.show()

def plot(sys, data, scale=1e-6, cmap='gnuplot', alpha=1, fig=None):
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
    fig: Maplotlib figure
        A plot is added to it if given. If not given, a new one is created and 
        displayed.
    """

    if not mpl_enabled:
        raise RuntimeError("matplotlib was not found, but is required "
                           "for plotsys()")

    xpts, ypts = sys.xpts / scale, sys.ypts / scale
    data = data.reshape(sys.ny, sys.nx)

    show = False
    if fig is None:
        fig = plt.figure()
        show = True

    ax = fig.add_subplot(111)
    ax.pcolor(xpts, ypts, data)
    ax.set_xlim(xmin=0, xmax=xpts[-1])
    ax.set_ylim(ymin=0, ymax=ypts[-1])
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    if show:
        plt.show()
