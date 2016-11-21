from sesame.utils import maps3D, integrator, extra_charges_path, get_xyz_from_s
from sim import *
import matplotlib.pyplot as plt
from sesame.observables import get_jn, get_jp, get_rr, get_n, get_p

# _, _, v = np.load('electrostatics.npy')
# _, _, efn, efp, v = np.load('bands.vapp_0.0.npy')
# maps3D(sys, v)

EGB = -0.3/sys.vt
ni = sys.ni[0]
nGB = sys.Nc[0] * np.exp(-sys.Eg[0]/2 + EGB)
pGB = sys.Nv[0] * np.exp(-sys.Eg[0]/2 - EGB)

from scipy.interpolate import InterpolatedUnivariateSpline as spline
x, y = sys.xpts, sys.ypts
# x = sys.xpts
# gx = [spline(x, sys.g[i*sys.nx:(i+1)*sys.nx]).integral(x[0], x[-1]) for\
#       i in range(sys.ny)]
# gtot = spline(y, gx).integral(y[0], y[-1])
# gtot = spline(sys.xpts, sys.g).integral(sys.xpts[0], sys.xpts[-1])

startGB = (20e-9, 2.5e-6, 0)
endGB = (2.8e-6, 2.5e-6, 0)
GBsites, X, xGB, yGB = extra_charges_path(sys, startGB, endGB)
x0, _, _ = get_xyz_from_s(sys, GBsites[0])
x1, _, _ = get_xyz_from_s(sys, GBsites[-1])

# sites_i = [sys.nx//2 + j*sys.nx for j in range(sys.ny)]
# sites_ip1 = [sys.nx//2+1 + j*sys.nx for j in range(sys.ny)]
# dl = sys.dx[sys.nx//2]
def integrator(sys, v, efn, efp, sites_i, sites_ip1, dl):
    # return the current in the x-direction, summed along the y-axis
    jn = get_jn(sys, efn, v, sites_i, sites_ip1, dl)
    jp = get_jp(sys, efp, v, sites_i, sites_ip1, dl)
    j = spline(sys.ypts, jn+jp).integral(sys.ypts[0], sys.ypts[-1])
    return j

def R(sys, efn, efp, v):
    u = []
    for j in range(sys.ny):
        s = [i + j*sys.nx for i in range(sys.nx)]
        n = get_n(sys, efn, v, s)
        p = get_p(sys, efp, v, s)
        r = get_rr(sys, n, p, sys.n1[s], sys.p1[s], sys.tau_e[s], sys.tau_h[s], s)
        sp = spline(sys.xpts, r)
        u.append(sp.integral(sys.xpts[0], sys.xpts[-1]))
    sp = spline(sys.ypts, u)
    return sp.integral(sys.ypts[0], sys.ypts[-1])



s = [10 + j*sys.nx for j in range(sys.ny)]
s = np.asarray(s)

S = float(osys.argv[1])*1e-2 / sys.Sc
Vapp = np.linspace(1, 10, 10)[:]
vv, u, uu = [], [], []
for vapp in Vapp:
    x, y, efn, efp, v = np.load('dark-bands.S_{1}.vapp_{0}.npy'.format(vapp, osys.argv[1]))
    # x, efn, efp, v = np.load('bands1D.dark.vapp_{0}.npy'.format(vapp))
    # vv.append(v[380 + sys.ny//2*sys.nx] - v[380])
    # I = integrator(sys, v, efn, efp, sites_i, sites_ip1, dl, integrate=True)
    # I = get_jn(sys, efn, v, 10, 11, sys.dx[10]) +  get_jp(sys, efp, v, 10, 11, sys.dx[10])
    I = integrator(sys, v, efn, efp, s, s+1, sys.dx[10])
    vv.append(-I)
    # print(vapp, I/gtot)

    p = get_p(sys, efp, v, GBsites)
    n = get_n(sys, efn, v, GBsites)
    sp = spline(sys.xpts[x0:x1+1], (n*p)/(n+p+nGB+pGB))
    r = sp.integral(sys.xpts[x0], sys.xpts[x1]) * S
    u.append(r)

    uu.append(R(sys, efn, efp, v))
    # s = [300 + j*sys.nx for j in range(sys.ny)]
    # plt.plot(y, efn[s], y, efp[s])
# np.savetxt('S_{0}.txt'.format(osys.argv[1]), np.column_stack((np.linspace(0, 40, 41), vv)))
# plt.plot(np.linspace(0, 40, 41), vv)
# plt.xlabel('$\mathregular{V_{app}}$')
# plt.ylabel('$\mathregular{V_{GB}}$')
np.savetxt('darkT100.dat', np.column_stack((Vapp, vv, u, uu)))
plt.show()
