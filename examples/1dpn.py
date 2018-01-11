import sesame
import numpy as np

L = 3e-6 # length of the system in the x-direction [m]

# Mesh
x = np.concatenate((np.linspace(0,1.2e-6, 100, endpoint=False), 
                    np.linspace(1.2e-6, L, 50)))

# Create a system
sys = sesame.Builder(x, input_length='m')

# Dictionary with the material parameters
CdTe = {'Nc':8e17*1e6, 'Nv':1.8e19*1e6, 'Eg':1.5, 'epsilon':9.4,
        'mu_e':100*1e-4, 'mu_h':100*1e-4, 'tau_e':10e-9, 'tau_h':10e-9}

# Add the material to the system
sys.add_material(CdTe)

junction = 50e-9 # extent of the junction from the left contact [m]
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

# Define Ohmic contacts
sys.contact_type('Ohmic', 'Ohmic')

# Define the surface recombination velocities for electrons and holes [m/s]
Sn_left, Sp_left, Sn_right, Sp_right = 1e50, 0, 0, 1e50
sys.contact_S(Sn_left, Sp_left, Sn_right, Sp_right)

# Define a function for the generation rate
phi = 1e21 # photon flux [1/(m^2 s)]
alpha = 2.3e6 # absorption coefficient [1/m]
f = lambda x: phi * alpha * np.exp(-alpha * x)
sys.generation(f)

solution = sesame.solve_equilibrium(sys)
# IV curve
voltages = np.linspace(0, 0.95, 40)
sesame.IVcurve(sys, voltages, solution, '1dpnIV')
