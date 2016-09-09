from sesame.observables import get_jn, get_jp
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np

def integrator(sys, v, efn, efp, sites_i, sites_ip1, dl, integrate=True):
    # return the current in the x-direction, summed along the y-axis
    jn = get_jn(sys, efn, v, sites_i, sites_ip1, dl)
    jp = get_jp(sys, efp, v, sites_i, sites_ip1, dl)
    j = jn+jp

    # integrate over y if 2d only
    if integrate:
        ypts = sys.ypts
        j = spline(ypts, j).integral(ypts[0], ypts[-1])
    else:
        j = j[0]
    return j

def current(files, vapplist, params, output=None, integrate=True, Voc=False, Pm=False):
    # output is printed on screen if set to None, otherwise data are stored in a
    # txt file with the name given by output
    if len(files) != len(vapplist):
        print('Lists of files and applied voltages need to have the same dimension!')
        exit(1)

    else:
        # compute normalization if non dark current
        gtot = 1
        if params.g.any() != 0:
            xpts = params.xpts
            ypts = params.ypts
            nx = len(xpts)
            g = params.g[:nx]
            gtot = spline(xpts, g).integral(xpts[0], xpts[-1]) * ypts[-1]
        # compute current
        c = []
        for fdx, f in enumerate(files):
            d = np.load(f)
            c.append(integrator(d[0], d[1], d[2], d[3], d[4], params, integrate) / (gtot))
        if type(output) == str:
            np.savetxt(output, np.column_stack((vapplist, c)))

        # compute Voc if asked
        if Voc == True:
            sp = spline(vapplist, np.asarray(c))
            Voc = brentq(lambda x: sp(x), 0, vapplist[-1])
            print('Open circuit voltage: ', Voc)

        # compute the point of maximum J * V
        if Pm == True:
            sp = spline(vapplist, np.asarray(c))
            dsp = sp.derivative()
            Vm = brentq(lambda x: dsp(x), 0, vapplist[-1])
            Jm = sp(Vm)
            print('Maximum power at Vm={0}, Jm={1}'.format(Vm, Jm))
        return c

def maps3D(sys, data, cmap='gnuplot', alpha=1):
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

def extra_charges_path(sys, start, end):
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

        if min(d1, d2, d3) == d1: # going up
            X.append(X[-1] + sys.dy[y])
            x, y = x, y+1
        elif xa < xb: # going right
            X.append(X[-1] + sys.dx[x])
            x, y = x+1, y
        elif xa > xb: # going left
            X.append(X[-1] + sys.dx[x-1])
            x, y = x-1, y
        s.append(x + y*sys.nx)
        xcoord.append(x)
        ycoord.append(y)
    return s, np.asarray(X), np.asarray(xcoord), np.asarray(ycoord)
