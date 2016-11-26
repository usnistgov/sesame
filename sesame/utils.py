from sesame.observables import get_jn, get_jp
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import numpy as np

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
    xa, ya = start[0], start[1]
    xb, yb = end[0], end[1]

    # reorder the points do that they are in ascending order
    if ya <= yb:
        ia, ja, ka = get_indices(sys, (xa, ya, 0))
        ib, jb, kb = get_indices(sys, (xb, yb, 0))
    else:
        ia, ja, ka = get_indices(sys, (xb, yb, 0))
        ib, jb, kb = get_indices(sys, (xa, ya, 0))

    distance = lambda x, y:\
        abs((yb-ya)*x - (xb-xa)*y + xb*ya - yb*xa)/\
            np.sqrt((yb-ya)**2 + (xb-xa)**2)

    def condition(x, y):
        if ia <= ib:
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

        if ia < ib: # overall direction is to the right
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
    xcoord = np.asarray(xcoord, dtype=int)
    ycoord = np.asarray(ycoord, dtype=int)

    # put everyting back to original order if inverted=True
    if xa > xb:
        s.reverse()
        X = np.flipud(X[-1] - X) * sys.xscale
        xcoord = np.flipud(xcoord)
        ycoord = np.flipud(ycoord)
    return s, X, xcoord, ycoord

def bulk_recombination_current(sys, efn, efp, v):
    """
    Compute the bulk recombination current.

    Parameters
    ----------
    sys: Builder
        The discretized system.
    efn: numpy array of floats
        One-dimensional array containing the electron quasi-Fermi level.
    efp: numpy array of floats
        One-dimensional array containing the hole quasi-Fermi level.
    v: numpy array of floats
        One-dimensional array containing the electrostatic potential.

    Returns
    -------
    JR: float
        The integrated bulk recombination.

    Warnings
    --------
    Not implemented in 3D.
    """
    u = []
    for j in range(sys.ny):
        # List of sites
        s = [i + j*sys.nx for i in range(sys.nx)]

        # Carrier densities
        n = get_n(sys, efn, v, s)
        p = get_p(sys, efp, v, s)

        # Recombination
        r = get_rr(sys, n, p, sys.n1[s], sys.p1[s], sys.tau_e[s], sys.tau_h[s], s)
        sp = spline(sys.xpts, r)
        u.append(sp.integral(sys.xpts[0], sys.xpts[-1]))
    if sys.ny == 1:
        JR = u[-1]
    if sys.ny > 1:
        sp = spline(sys.ypts, u)
        JR = sp.integral(sys.ypts[0], sys.ypts[-1])
    return JR
 
def full_current(sys, efn, efp, v):
    """
    Compute the steady state current.

    Parameters
    ----------
    sys: Builder
        The discretized system.
    efn: numpy array of floats
        One-dimensional array containing the electron quasi-Fermi level.
    efp: numpy array of floats
        One-dimensional array containing the hole quasi-Fermi level.
    v: numpy array of floats
        One-dimensional array containing the electrostatic potential.

    Returns
    -------
    JR: float
        The integrated bulk recombination.

    Warnings
    --------
    Not implemented in 3D.
    """
    # Define the sites between which computing the currents
    sites_i = [sys.nx//2 + j*sys.nx for j in range(sys.ny)]
    sites_ip1 = [sys.nx//2+1 + j*sys.nx for j in range(sys.ny)]
    # And the corresponding lattice dimensions
    dl = sys.dx[sys.nx//2]

    # Compute the electron and hole currents
    jn = get_jn(sys, efn, v, sites_i, sites_ip1, dl)
    jp = get_jp(sys, efp, v, sites_i, sites_ip1, dl)

    if sys.ny == 1:
        j = jn + jp
    if sys.ny > 1:
        # Interpolate the results and integrate over the y-direction
        j = spline(sys.ypts, jn+jp).integral(sys.ypts[0], sys.ypts[-1])

    return j
