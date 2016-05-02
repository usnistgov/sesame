from sesame.observables import get_jn, get_jp
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np


def integrator(xpts, ypts, v, efn, efp, params, integrate=True):
    # return the current in the x-direction, summed along the y-axis
    nx = len(xpts)
    idx = nx//2 # let's compute the current in the middle of the system
    dx = xpts[1:] - xpts[:-1]
    mu = params.mu
    c = []
    for j in range(len(ypts)):
        s = idx + j*nx
        # jn = mu[s] * get_jn(efn[s],efn[s+1],v[s],v[s+1],dx[idx],params)
        # jp = mu[s] * get_jp(efp[s],efp[s+1],v[s],v[s+1],dx[idx],params)
        jn = get_jn(efn[s],efn[s+1],v[s],v[s+1],dx[idx],params)
        jp = get_jp(efp[s],efp[s+1],v[s],v[s+1],dx[idx],params)
        c.append(jn+jp)

    # integrate over y if 2d only
    if integrate:
        c = spline(ypts, c).integral(ypts[0], ypts[-1])
    else:
        c = c[0]
    return c

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
            # gtot=1
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

def maps3D(xpts, ypts, data, cmap='gnuplot', alpha=1):
    nx, ny = len(xpts), len(ypts)
    data_xy = data.reshape(ny, nx).T
    X, Y = np.meshgrid(xpts, ypts)
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(1,1,1, projection='3d')
    Z = data_xy.T
    ax.plot_surface(X, Y, Z,  alpha=alpha, cmap=cmap)
    ax.mouse_init(rotate_btn=1, zoom_btn=3)
    plt.show()

def get_coordinates(x, y, params, site=False):
    # Return the discrete set of coordinates based on continous  coordinates
    # If site is True, return the site number instead
    xpts, ypts = params.xpts, params.ypts
    nx, ny = len(xpts), len(ypts)
    x, y = nx-len(xpts[xpts >= x]), ny-len(ypts[ypts >= y])
    if site:
        s = x + nx * y
        return s
    else:
        return x, y

def get_indices(x, y, xpts, ypts):
    # Return the discrete set of coordinates based on continous  coordinates
    # If site is True, return the site number instead
    nx, ny = len(xpts), len(ypts)
    x, y = nx-len(xpts[xpts >= x]), ny-len(ypts[ypts >= y])
    return x, y
