import sesame
import numpy as np


# dimensions of the system
Lx = 3e-6 #[m]
Ly = 5e-6 #[m]
# extent of the junction from the left contact [m]
junction = 10e-9 

# Mesh
x = np.concatenate((np.linspace(0,1.2e-6, 100, endpoint=False), 
                    np.linspace(1.2e-6, Lx, 50)))

sys = sesame.Builder(x)

def region(pos):
    x = pos
    return x < junction

# Add the donors
nD = 1e17 * 1e6 # [m^-3]
sys.add_donor(nD, region)

# Add the acceptors
region2 = lambda pos: 1 - region(pos)
nA = 1e15 * 1e6 # [m^-3]
sys.add_acceptor(nA, region2)

# Use perfectly selective contacts
Sn_left, Sp_left, Sn_right, Sp_right = 1e10, 0, 0, 1e10
sys.contacts(Sn_left, Sp_left, Sn_right, Sp_right)

# Dictionary with the material parameters
reg1 = {'Nc':8e17*1e6, 'Nv':1.8e19*1e6, 'Eg':1.5, 'epsilon':9.4,
        'mu_e':200*1e-4, 'mu_h':200*1e-4, 'tau_e':10e-9, 'tau_h':10e-9, 
        'RCenergy':0, 'band_offset':0}
# Add the material to the system
sys.add_material(reg1)

sys.finalize()

# Solve the Poisson equation
v_left  = np.log(sys.rho[0]/sys.Nc[0])
v_right = -sys.Eg[0] - np.log(abs(sys.rho[sys.nx-1])/sys.Nv[sys.nx-1])

v = np.linspace(v_left, v_right, sys.nx)

print('solving...')
res = sesame.poisson_solver(sys, v, iterative=True)
import matplotlib.pyplot as plt
plt.plot(x*1e6, res[:sys.nx])
plt.show()
res = sesame.ddp_solver(sys, [0*v, 0*v, res], iterative=True, maxiter=2)
