from scipy.interpolate import InterpolatedUnivariateSpline as spline
import numpy as np
from . import observables

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

def line_defects_sites(sys, location):
    """
    Get sites and coordinates of the locations containing additional charges
    distributed on a line.

    Parameters
    ----------

    sys: Builder
        The discretized system.
    location: list of two array_like coordinates [(x1, y1), (x2, y2)]
        Coordinates of the two points of the line containing additional
        charges in [m].

    Returns
    -------

    s: numpy array of integers
        Sites numbers.
    X: numpy array of floats
        Incremental sum of the lattice size between sites.
    iccord: list of integers
        Indices of the x-coordinates of the sites on the discretized lattice.
    jccord: list of integers
        Indices of the y-coordinates of the sites on the discretized lattice.

    Notes
    -----
    This only works in 2D.
    """

    x1, y1 = location[0]
    x2, y2 = location[1]

    sites, X, icoord, jcoord, _ = Bresenham2d(sys, (x1, y1, 0), (x2, y2, 0))
    sites = np.asarray(sites)
    X = np.asarray(X)
    return sites, X, icoord, jcoord

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

    x = sys.xpts / sys.scaling.length
    y = sys.ypts / sys.scaling.length
    u = []
    for j in range(sys.ny):
        # List of sites
        s = [i + j*sys.nx for i in range(sys.nx)]

        # Carrier densities
        n = observables.get_n(sys, efn, v, s)
        p = observables.get_p(sys, efp, v, s)

        # Recombination
        r = observables.get_rr(sys, n, p, sys.n1[s], sys.p1[s], sys.tau_e[s], sys.tau_h[s], s)
        sp = spline(x, r)
        u.append(sp.integral(x[0], x[-1]))
    if sys.ny == 1:
        JR = u[-1]
    if sys.ny > 1:
        sp = spline(y, u)
        JR = sp.integral(y[0], y[-1])
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
    jn = observables.get_jn(sys, efn, v, sites_i, sites_ip1, dl)
    jp = observables.get_jp(sys, efp, v, sites_i, sites_ip1, dl)

    if sys.ny == 1:
        j = jn + jp
    if sys.ny > 1:
        # Interpolate the results and integrate over the y-direction
        y = sys.ypts / sys.scaling.length
        j = spline(y, jn+jp).integral(y[0], y[-1])

    return j


def Bresenham2d(system, p1, p2):
    # Compute a digital line contained in the plane (x,y) and passing by the
    # points p1 and p2.
    i1, j1, k1 = get_indices(system, p1)
    i2, j2, k2 = get_indices(system, p2)

    Dx = abs(i2 - i1)    # distance to travel in X
    Dy = abs(j2 - j1)    # distance to travel in Y

    incx = 0
    if i1 < i2:
        incx = 1           # x will increase at each step
    elif i1 > i2:
        incx = -1          # x will decrease at each step

    incy = 0
    if j1 < j2:
        incy = 1           # y will increase at each step
    elif j1 > j2:
        incy = -1          # y will decrease at each step

    # take the numerator of the distance of a point to the line
    error = lambda x, y: abs((p2[1]-p1[1])*x - (p2[0]-p1[0])*y + p2[0]*p1[1] - p2[1]*p1[0])

    i, j = i1, j1
    sites = [i + j*system.nx + k1*system.nx*system.ny]
    X = [0]
    icoord = [i]
    jcoord = [j]
    kcoord = [k1]
    for _ in range(Dx + Dy):
        e1 = error(system.xpts[i], system.ypts[j+incy])
        e2 = error(system.xpts[i+incx], system.ypts[j])
        if incx == 0:
            condition = e1 <= e2 # for lines x = constant
        else:
            condition = e1 < e2
        if condition:
            j += incy
            X.append(X[-1] + system.dy[j])
        else:
            i += incx
            X.append(X[-1] + system.dx[i])
        sites.append(i + j*system.nx + k1*system.nx*system.ny)
        icoord.append(i)
        jcoord.append(j)
        kcoord.append(k1)
    return sites, X, icoord, jcoord, kcoord

def check_plane(P1, P2, P3, P4):
    # check if plane is within what can be handled
    msg = "Acceptable planes are rectangles with at least one edge parallel " +\
          "to either x, or y or z. The rectangles must be defined by two lines " +\
          "perpendicular to the z-axis."

    # vectors within the plane
    v1 = P2 - P1
    v2 = P3 - P1
    vperp = np.cross(v1, v2)

    check = True

    # 1. check if the two lines are perpendicular to th z-axis
    if np.dot(v1, (0,0,1)) != 0 and np.dot(v3, (0,0,1)):
        check = False
        print("The lines defining the plane defects defined by the four points "\
            + "{0}, {1}, {2}, {3}".format(P1, P2, P3, P4) +\
            " are not perpendicular to the z-axis.")

    # 2. check if the plane is a rectangle
    center = (P1 + P2 + P3 + P4) / 4.  # compute center of the figure
    d1 = np.linalg.norm(P1 - center)
    d2 = np.linalg.norm(P2 - center)
    d3 = np.linalg.norm(P3 - center)
    d4 = np.linalg.norm(P4 - center)

    if not all(abs(d-d1) < 1e-15 for d in (d2, d3, d4)):
        check = False
        print("The plane defects defined by the four points " +\
              "{0}, {1}, {2}, {3}".format(P1, P2, P3, P4) +\
              " is not a rectangle.")

    # 3. check if the plane is parallel to at least one axis
    c = abs(np.dot(vperp, (1,0,0))) > 1e-30 and\
        abs(np.dot(vperp, (0,1,0))) > 1e-30 and\
        abs(np.dot(vperp, (0,0,1))) > 1e-30
    if c:
        check = False
        print("The plane defects defined by the four points " +\
              "{0}, {1}, {2}, {3}".format(P1, P2, P3, P4) +\
              " is not a parallel to at least one main axis.")

    if not check:
        print(msg)
        exit(1)

def plane_defects_sites(sys, location):
    """
    Get sites and coordinates of the locations containing additional charges
    distributed on a plane.

    Parameters
    ----------

    sys: Builder
        The discretized system.
    location: list of four array_like coordinates [(x1, y1, z1), (x2, y2, z2), (x3, y3, z3), (x4, y4, z4)]
        The coordinates in [m] define a plane of defects in 3D. The first two
        coordinates define a line that must be parallel to the line defined by
        the last two points.

    Returns
    -------

    s: numpy array of integers
        Sites numbers.
    xccord: numpy array of floats
        Grid of the x-coordinates of the sites [m].
    yccord: numpy array of floats
        Grid of the y-coordinates of the sites [m].

    Notes
    -----
    This only works in 3D.
    """

    # transform points into numpy arrays if not the case
        
    P1, P2, P3, P4 = [np.asarray(P) for P in location]

    # check plane first
    check_plane(P1, P2, P3, P4)

    # points indices on the grid
    i1, j1, k1 = get_indices(sys, P1)
    i2, j2, k2 = get_indices(sys, P2)
    # check if the two lines are in the same direction, if not, flip the second line
    if np.dot(P2 - P1, P4 - P3) > 0:
        i3, j3, k3 = get_indices(sys, P3)
    else:
        i3, j3, k3 = get_indices(sys, P4)

    ## vector perpendicular to the plane
    A, B, C = np.cross(P2 - P1, P3 - P1)
    D = -A*P1[0] - B*P1[1] - C*P1[2]

    error = lambda x, y, z: abs(A*x + B*y + C*z + D)
    
    Dx = abs(i3 - i1)    # distance to travel in X
    Dy = abs(j3 - j1)    # distance to travel in Y
    Dz = abs(k3 - k1)    # distance to travel in Z
    travel = Dx + Dy + Dz

    # increment in x
    incx = 0
    if i1 < i3:
        incx = 1
    elif i1 > i3:
        incx = -1
    # increment in y
    incy = 0
    if j1 < j3:
        incy = 1
    elif j1 > j3:
        incy = -1
    # increment in z
    incz = 0
    if k1 < k3:
        incz = 1
    elif k1 > k3:
        incz = -1
       
    sites, xcoord, ycoord, zcoord = [], [], [], []

    s, _, ic, jc, kc = Bresenham2d(sys, P1, P2)

    sites.extend(s)
    xcoord.append(sys.xpts[ic])
    ycoord.append(sys.ypts[jc])
    zcoord.append(sys.zpts[kc])

    for _ in range(travel-1):
        # find the coordinates of the next line
        e1 = error(sys.xpts[i1+incx], sys.ypts[j1], sys.zpts[k1])
        e2 = error(sys.xpts[i1], sys.ypts[j1+incy], sys.zpts[k1])
        e3 = error(sys.xpts[i1], sys.ypts[j1], sys.zpts[k1+incz])

        if incz == 0:
            if e1 < e2:
                i1, j1, k1 = i1 + incx, j1, k1
                i2, j2, k2 = i2 + incx, j2, k2
            else:
                i1, j1, k1 = i1, j1 + incy, k1
                i2, j2, k2 = i2, j2 + incy, k2

        if incy == 0 and incx != 0:
            if e1 < e3:
                i1, j1, k1 = i1 + incx, j1, k1
                i2, j2, k2 = i2 + incx, j2, k2
            else:
                i1, j1, k1 = i1, j1, k1 + incz
                i2, j2, k2 = i2, j2, k2 + incz

        if incx == 0 and incy != 0:
            if e2 < e3:
                i1, j1, k1 = i1, j1 + incy, k1
                i2, j2, k2 = i2, j2 + incy, k2
            else:
                i1, j1, k1 = i1, j1, k1 + incz
                i2, j2, k2 = i2, j2, k2 + incz

        if incx == 0 and incy == 0:
            i1, j1, k1 = i1, j1, k1 + incz
            i2, j2, k2 = i2, j2, k2 + incz

        x1, x2 = sys.xpts[i1], sys.xpts[i2]
        y1, y2 = sys.ypts[j1], sys.ypts[j2]
        z1, z2 = sys.zpts[k1], sys.zpts[k2]

        s, _, ic, jc, kc = Bresenham2d(sys, (x1, y1, z1), (x2, y2, z2))

        sites.extend(s)
        xcoord.append(sys.xpts[ic])
        ycoord.append(sys.ypts[jc])
        zcoord.append(sys.zpts[kc])

    xcoord = np.asarray(xcoord)
    ycoord = np.asarray(ycoord)
    zcoord = np.asarray(zcoord)
    return sites, xcoord, ycoord, zcoord
