import sesame
import numpy as np
import scipy.io
from scipy.io import savemat
from matplotlib import pyplot as plt


t1 = 25*1e-7    # thickness of CdS
t2 = 4*1e-4     # thickness of CdTe

# Heterojunctions require dense mesh near the interface
dd = 1e-7   # 2*dd is the distance over which mesh is refined
# Define the mesh
x = np.concatenate((np.linspace(0, dd, 10, endpoint=False),                        # L contact interface
                    np.linspace(dd, t1-dd, 50, endpoint=False),                    # material 1
                    np.linspace(t1 - dd, t1 + dd, 10, endpoint=False),             # interface 1
                    np.linspace(t1 + dd, (t1+t2) - dd, 100, endpoint=False),       # material 2
                    np.linspace((t1+t2) - dd, (t1+t2), 10)))                       # R contact interface

# Build system
sys = sesame.Builder(x)

# CdS material dictionary
CdS = {'Nc': 2.2e18, 'Nv':1.8e19, 'Eg':2.4, 'epsilon':10, 'Et': 0,
        'mu_e':100, 'mu_h':25, 'tau_e':1e-8, 'tau_h':1e-13,
        'affinity': 4.}
# CdTe material dictionary
CdTe = {'Nc': 8e17, 'Nv': 1.8e19, 'Eg':1.5, 'epsilon':9.4, 'Et': 0,
        'mu_e':320, 'mu_h':40, 'tau_e':5e-9, 'tau_h':5e-9,
        'affinity': 3.9}

# CdS region
region1 = lambda x: x<=t1
# CdTe region
region2 = lambda x: x>t1

# Add the material to the system
sys.add_material(CdS, region1)     # adding CdS
sys.add_material(CdTe, region2)     # adding CdTe

# Add the donors
nD = 1e17  # donor density [cm^-3]
sys.add_donor(nD, region1)
# Add the acceptors
nA = 1e15  # acceptor density [cm^-3]
sys.add_acceptor(nA, region2)

# Define contacts: CdS contact is Ohmic, CdTe contact is Schottky
Lcontact_type, Rcontact_type = 'Ohmic', 'Schottky'
Lcontact_workFcn, Rcontact_workFcn = 0, 5.0   # Lcontact work function irrelevant because L contact is Ohmic
# Add the contacts
sys.contact_type(Lcontact_type, Rcontact_type, Lcontact_workFcn, Rcontact_workFcn)

# Define the surface recombination velocities for electrons and holes [m/s]
Scontact = 1.16e7  # [cm/s]
# non-selective contacts
Sn_left, Sp_left, Sn_right, Sp_right = Scontact, Scontact, Scontact, Scontact
# This function specifies the simulation contact recombination velocity
sys.contact_S(Sn_left, Sp_left, Sn_right, Sp_right)

# First find the equilibrium solution
result = sesame.solve_equilibrium(sys)

# Add illumination
phi0 = 1e17     # incoming flux [1/(cm^2 sec)]
alpha = 2.3e4   # absorbtion coefficient [1/cm]
# Define a function for illumination profile
f = lambda x: phi0*alpha*np.exp(-x*alpha)   # f is an "inline" function
# This function adds generation to the simulation
sys.generation(f)


# Define prefix of output files
filename = 'CdS_CdTe'
# Specify the applied voltage values
voltages = np.linspace(0,1,21)
# Perform J-V calculation
j = sesame.IVcurve(sys, voltages, result, filename)
j = j * sys.scaling.current

np.save('jv1d_hetero', result)

import matplotlib.pyplot as plt
plt.plot(voltages, j, '-o')
plt.xlabel('Voltage [V]')
plt.ylabel('Current [A/cm^2]')
plt.grid()
plt.show()

