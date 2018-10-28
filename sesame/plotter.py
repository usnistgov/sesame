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


def plot_grid(sys, fig=None):
    """
    Plot the grid of a 2D system.

    Parameters
    ----------
    sys: Builder
        The discretized system.
    fig: Maplotlib figure
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

    for xpt in sys.xpts:
        ax.plot([xpt,xpt],[sys.ypts[0],sys.ypts[-1]],color='k',linewidth=.5)
    for ypt in sys.ypts:
        ax.plot([sys.xpts[0],sys.xpts[-1]],[ypt,ypt],color='k',linewidth=.5)


    if show:
        plt.show()


def plot_line_defects(sys, ls='-o', fig=None):
    """
    Plot the sites containing additional charges located on lines in 2D. The
    length scale of the graph is 1 micrometer by default.

    Parameters
    ----------
    sys: Builder
        The discretized system.
    ls: string
        Line style of the plotted paths.
    fig: Maplotlib figure
        A plot is added to it if given. If not given, a new one is created and a
        figure is displayed.
    """
    if not mpl_enabled:
        raise RuntimeError("matplotlib was not found, but is required "
                           "for plotting.")

    show = False
    if fig is None:
        fig = plt.figure()
        show = True
    # add axis to figure
    try:
        ax = fig.axes[0]
    except Exception:
        ax = fig.add_subplot(111)

    for c in sys.defects_list:
        xa, ya = c.location[0]
        xb, yb = c.location[1]

        _, _, xcoord, ycoord = utils.Bresenham(sys, (xa, ya,0), (xb,yb,0))

        # plot the path of added charges
        ax.plot(sys.xpts[xcoord], sys.ypts[ycoord], ls)

    if sys.input_length == 'm':
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
    if sys.input_length == 'cm':
        ax.set_xlabel('x [cm]')
        ax.set_ylabel('y [cm]')
       
    ax.set_xlim(xmin=0, xmax=sys.xpts[-1])
    ax.set_ylim(ymin=0, ymax=sys.ypts[-1])

    if show:
        plt.show()

def plot(sys, data, cmap='gnuplot', alpha=1, fig=None, title=''):
    """
    Plot a 2D map of a parameter (like mobility) across the system.

    Parameters
    ----------

    sys: Builder
        The discretized system.
    data: numpy array
        One-dimensional array of data with size equal to the size of the system.
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
                           "for this feature.")

    if sys.input_length == 'cm':
        scale = 1e-4
    elif sys.input_length == 'm':
        scale = 1e-6
    xpts, ypts = sys.xpts / scale, sys.ypts / scale
    data = data.reshape(sys.ny, sys.nx)

    show = False
    if fig is None:
        fig = plt.figure()
        show = True

    ax = fig.add_subplot(111)
    p = ax.pcolor(xpts, ypts, data)
    cbar = fig.colorbar(p, ax=ax)

    ax.set_xlim(xmin=0, xmax=xpts[-1])
    ax.set_ylim(ymin=0, ymax=ypts[-1])
    ax.set_xlabel(r'x [$\mathregular{\mu m}$]')
    ax.set_ylabel(r'y [$\mathregular{\mu m}$]')
    ax.set_title(title)

    if show:
        plt.show()
