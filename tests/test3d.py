import sesame
import numpy as np


# dimensions of the system
Lx = 3e-6 #[m]
Ly = 5e-6 #[m]
Lz = 2e-6 #[m]
# extent of the junction from the left contact [m]
junction = 10e-9 

# Mesh
x = np.concatenate((np.linspace(0,1.2e-6, 100, endpoint=False), 
                    np.linspace(1.2e-6, Lx, 20)))
y = np.linspace(0, Ly, 50)
z = np.linspace(0, Lz, 20)

sys = sesame.Builder(x, y, z)

def region(pos):
    x, y, z = pos
    return x < junction

# Add the donors
nD = 1e17 * 1e6 # [m^-3]
sys.add_donor(nD, region)

# Add the acceptors
region2 = lambda pos: 1 - region(pos)
nA = 1e15 * 1e6 # [m^-3]
sys.add_acceptor(nA, region2)

# Use perfectly selective contacts
Sn_left, Sp_left, Sn_right, Sp_right = 1e50, 0, 0, 1e50
sys.contacts(Sn_left, Sp_left, Sn_right, Sp_right)

# Dictionary with the material parameters
reg1 = {'Nc':8e17*1e6, 'Nv':1.8e19*1e6, 'Eg':1.5, 'epsilon':9.4,
        'mu_e':200*1e-4, 'mu_h':200*1e-4, 'tau_e':10e-9, 'tau_h':10e-9, 
        'RCenergy':0, 'band_offset':0}
# Add the material to the system
sys.add_material(reg1)

# gap state characteristics
S = 1e5 * 1e-2           # trap recombination velocity [m/s]
E = -0.25                # energy of gap state (eV) from midgap
N = 2e14 * 1e4           # defect density [1/m^2]

# Specify the two points that make the line containing additional charges
p1 = (1e-6, .5e-6, 1e-6)    #[m]
p2 = (2.9e-6, .5e-6, 1e-6)  #[m]

q1 = (1.0e-6, 4.5e-6, 1e-9) #[m]
q2 = (2.9e-6, 4.5e-6, 1e-9) #[m]

# Pass the information to the system
sys.add_plane_defects([p1, p2, q1, q2], E, N, S)


# sesame.plot_plane_defects(sys)

# Solve the Poisson equation
v_left  = np.log(sys.rho[0]/sys.Nc[0])
v_right = -sys.Eg[0] - np.log(abs(sys.rho[sys.nx-1])/sys.Nv[sys.nx-1])

v = np.empty((sys.nx,), dtype=float) 
v[:sys.nx] = np.linspace(v_left, v_right, sys.nx)
v = np.tile(v, sys.ny*sys.nz) # replicate the guess in the y-direction

print('solving...')
res = sesame.poisson_solver(sys, v, iterative=True)
# np.save('electrostatic', res)
# import matplotlib.pyplot as plt
# plt.plot(x*1e6, res[:sys.nx])
# plt.show()
print()
res = sesame.ddp_solver(sys, [0*v, 0*v, res], iterative=True, eps=1e-4)
