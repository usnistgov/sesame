import sesame
import numpy as np
import cProfile
import re


# dimensions of the system
Lx = 3e-6 #[m]
Ly = 5e-6 #[m]
Lz = 2e-6 #[m]
# extent of the junction from the left contact [m]
junction = 10e-9 

# Mesh
x = np.concatenate((np.linspace(0,1.2e-6, 5, endpoint=False), 
                    np.linspace(1.2e-6, Lx, 5)))
y = np.linspace(0, Ly, 100)
z = np.linspace(0, Lz, 100)

sys = sesame.Builder(x, y, z)

def region(pos):
    x, y, z = pos
    return x < junction

# Add the donors
nD = 1e17 * 1e6 # [m^-3]
print("add donor")
sys.add_donor(nD, region)

# Add the acceptors
region2 = lambda pos: 1 - region(pos)
nA = 1e15 * 1e6 # [m^-3]
print('add acceptor')
sys.add_acceptor(nA, region2)

# Use perfectly selective contacts
Sn_left, Sp_left, Sn_right, Sp_right = 1e50, 0, 0, 1e50
print('contacts')
sys.contacts(Sn_left, Sp_left, Sn_right, Sp_right)

# Dictionary with the material parameters
reg1 = {'Nc':8e17*1e6, 'Nv':1.8e19*1e6, 'Eg':1.5, 'epsilon':9.4,
        'mu_e':200*1e-4, 'mu_h':200*1e-4, 'tau_e':10e-9, 'tau_h':10e-9, 
        'RCenergy':0, 'band_offset':0}
print("add material")
# Add the material to the system
sys.add_material(reg1)

# gap state characteristics
S = 1e5 * 1e-2           # trap recombination velocity [m/s]
E = -0.25                # energy of gap state (eV) from midgap
N = 2e14 * 1e4           # defect density [1/m^2]

# Specify the two points that make the line containing additional charges
p1 = (1e-6, .5e-6, 1e-6)   #[m]
p2 = (3.0e-6, .5e-6, 1e-6)  #[m]

q1 = (1.0e-6, 4.5e-6, 1e-9)   #[m]
q2 = (3.0e-6, 4.5e-6, 1e-9)  #[m]


# print('add plane defects')
# Pass the information to the system
# sys.add_plane_defects([p1, p2, q1, q2], E, N, S)


sys.finalize()


import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
def plot_coo_matrix(m):
    if not isinstance(m, coo_matrix):
        m = coo_matrix(m)
    fig = plt.figure()
    ax = fig.add_subplot(111, axisbg='black')
    ax.plot(m.col, m.row, 's', color='white', ms=1)
    ax.set_xlim(0, m.shape[1])
    ax.set_ylim(0, m.shape[0])
    ax.set_aspect('equal')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    return ax



# sesame.plot_plane_defects(sys)
v = np.linspace(-5, -40, sys.nx)
v = np.tile(v, sys.ny*sys.nz)

from sesame.getF3 import getF
from sesame.jacobian3 import getJ

f = getF(sys, v, 0*v, v)
J = getJ(sys, v, 0*v, v, with_mumps=True)

plt.spy(J)
plt.show()
# ax = plot_coo_matrix(J)
# ax.figure.show()

# print("solving...")
# from sesame.mumps import spsolve
# spsolve(J, -f)
