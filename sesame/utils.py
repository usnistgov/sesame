from sesame.observables import get_jn, get_jp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np

def plot(sys, ls='-o'):
    """
    Plot the sites containing additional charges.

    Parameters
    ----------
    sys: Builder
        The discretized system.
    ls: string
        Line style of the plotted paths.
    """

    for c in sys.charges:
        xa, ya, za = get_indices(sys, (c.xa, c.ya, c.za))
        xb, yb, zb = get_indices(sys, (c.xb, c.yb, c.zb))

        # find the sites closest to the straight line defined by
        # (xa,ya,za) and (xb,yb,zb) and the associated dl       
        distance = lambda x, y:\
            abs((c.yb-c.ya)*x - (c.xb-c.xa)*y + c.xb*c.ya - c.yb*c.xa)/\
                np.sqrt((c.yb-c.ya)**2 + (c.xb-c.xa)**2)

        def condition(x, y):
            if xa <= xb:
                return x <= xb and y <= yb and x < sys.nx-1 and y < sys.ny-1
            else:
                return x >= xb and y <= yb and x > 1 and y < sys.ny-1

        x, y = xa, ya
        xcoord, ycoord = [xa], [ya]
        while condition(x, y):
            # distance between the point above (x,y) and the segment
            d1 = distance(sys.xpts[x], sys.ypts[y+1])
            # distance between the point right of (x,y) and the segment
            d2 = distance(sys.xpts[x+1], sys.ypts[y])
            # distance between the point left of (x,y) and the segment
            d3 = distance(sys.xpts[x-1], sys.ypts[y])

            if xa < xb: # overall direction is to the right
                if d1 < d2:
                    x, y = x, y+1
                else:
                    x, y = x+1, y
            else: # overall direction is to the left
                if d1 < d3:
                    x, y = x, y+1
                else:
                    x, y = x-1, y
            xcoord.append(x)
            ycoord.append(y)

        # plot the path of added charges
        sc = sys.xscale*1e6
        plt.plot(sys.xpts[xcoord]*sc, sys.ypts[ycoord]*sc, ls)

    plt.xlim(xmin=0, xmax=sys.xpts[-1]*sc)
    plt.ylim(ymin=0, ymax=sys.ypts[-1]*sc)
    plt.show()

def maps3D(sys, data, cmap='gnuplot', alpha=1):
    """
    Plot a 3D map of data across the system.

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
    """

    xpts, ypts = sys.xpts * sys.xscale * 1e6, sys.ypts * sys.xscale * 1e6
    nx, ny = len(xpts), len(ypts)
    data_xy = data.reshape(ny, nx).T
    X, Y = np.meshgrid(xpts, ypts)
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(1,1,1, projection='3d')
    Z = data_xy.T
    ax.plot_surface(X, Y, Z,  alpha=alpha, cmap=cmap)
    ax.mouse_init(rotate_btn=1, zoom_btn=3)
    plt.show()

def get_indices(sys, p, site=False):
    # Return the indices of continous coordinates on the discrete lattice
    # If site is True, return the site number instead
    # Warning: for x=Lx, the site index will be the one before the last one, and
    # the same goes for y=Ly and z=Lz.
    # p: list containing x,y,z coordinates, use zeros for unused dimensions

    x, y, z = p
    xpts, ypts, zpts = sys.xpts, sys.ypts, sys.zpts
    nx = len(xpts)
    x = nx-len(xpts[xpts >= x])
    s = x

    if ypts is not None:
        ny = len(ypts)
        y = ny-len(ypts[ypts >= y])
        s += nx*y

    if zpts is not None:
        nz = len(zpts)
        z = nz-len(zpts[zpts >= z])
        s += nx*ny*z

    if site:
        return s
    else:
        return x, int(y), int(z)

def get_xyz_from_s(sys, site):
    nx, ny, nz = sys.nx, sys.ny, sys.nz
    k = site // (nx * ny) * (nz > 1)
    j = (site - k*nx*ny) // nx * (ny > 1)
    i = site - j*nx - k*nx*ny
    return i, j, k

def get_dl(sys, sites):
    sa, sb = sites
    xa, ya, za = get_xyz_from_s(sys, sa)
    xb, yb, zb = get_xyz_from_s(sys, sb)

    if xa != xb: dl = sys.dx[xa]
    if ya != yb: dl = sys.dy[ya]
    if za != zb: dl = sys.dz[za]

    return dl

    return s, np.asarray(X), np.asarray(xcoord), np.asarray(ycoord)
def extra_charges_path(sys, start, end):
    """
    Get sites and coordinates of the locations containing additional charges.

    Parameters
    ----------

    sys: Builder
        The discretized system.
    start: Tuple (x, y, z)
        Coordinates of the first point of the line containing additional
        charges in [m].
    end: Tuple (x, y, z)
        Coordinates of the last point of the line containing additional
        charges [m].

    Returns
    -------

    s: numpy array of integers
        Sites numbers.
    X: numpy array of floats
        Incremental sum of the lattice size between sites.
    xccord: numpy array of floats
        x-coordinates of the sites on the discretized lattice.
    yccord: numpy array of floats
        y-coordinates of the sites on the discretized lattice.

    Notes
    -----
    This only works in 2D.
    """
    # Return the path and the sites
    xa, ya = start[0]/sys.xscale, start[1]/sys.xscale
    xb, yb = end[0]/sys.xscale, end[1]/sys.xscale
    
    ia, ja, _ = get_indices(sys, [xa, ya, 0])
    ib, jb, _ = get_indices(sys, [xb, yb, 0])

    distance = lambda x, y:\
        abs((yb-ya)*x - (xb-xa)*y + xb*ya - yb*xa)/\
            np.sqrt((yb-ya)**2 + (xb-xa)**2)

    def condition(x, y):
        if xa <= xb:
            return x <= ib and y <= jb and x < sys.nx-1 and y < sys.ny-1
        else:
            return x >= ib and y <= jb and x > 1 and y < sys.ny-1
                        
    xcoord, ycoord = [], []
    s = [ia + ja*sys.nx]
    X = [0]
    x, y = ia, ja
    while condition(x, y):
        # distance between the point above (x,y) and the segment
        d1 = distance(sys.xpts[x], sys.ypts[y+1])
        # distance between the point right of (x,y) and the segment
        d2 = distance(sys.xpts[x+1], sys.ypts[y])
        # distance between the point left of (x,y) and the segment
        d3 = distance(sys.xpts[x-1], sys.ypts[y])

        if xa < xb: # overall direction is to the right
            if d1 < d2:
                x, y = x, y+1
                X.append(X[-1] + sys.dy[y])
            else:
                x, y = x+1, y
                X.append(X[-1] + sys.dx[x])
        else: # overall direction is to the left
            if d1 < d3:
                x, y = x, y+1
                X.append(X[-1] + sys.dy[y])
            else:
                x, y = x-1, y
                X.append(X[-1] + sys.dx[x-1])
        s.append(x + y*sys.nx)
        xcoord.append(x)
        ycoord.append(y)
        
        X = np.asarray(X)
        xcoord = np.asarray(xcoord)
        ycoord = np.asarray(ycoord)
    return s, X, xcoord, ycoord
