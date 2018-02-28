import sesame
import numpy as np

L = 3e-4 # length of the system in the x-direction [cm]

# Mesh
x = np.concatenate((np.linspace(0,1.2e-4, 100, endpoint=False),
                    np.linspace(1.2e-4, L, 50)))

# Create a system
sys = sesame.Builder(x)

# Dictionary with the material parameters
CdTe = {'Nc':8e17, 'Nv':1.8e19, 'Eg':1.5, 'epsilon':9.4,
        'mu_e':100, 'mu_h':100, 'tau_e':10e-9, 'tau_h':10e-9}

# Add the material to the system
sys.add_material(CdTe)

junction = 50e-7 # extent of the junction from the left contact [m]
def n_region(pos):
    x = pos
    return x < junction

def p_region(pos):
    x = pos
    return x >= junction

# Add the donors
nD = 1e17 # [cm^-3]
sys.add_donor(nD, n_region)

# Add the acceptors
nA = 1e15 # [cm^-3]
sys.add_acceptor(nA, p_region)

# Define Ohmic contacts
sys.contact_type('Ohmic', 'Ohmic')

# Define the surface recombination velocities for electrons and holes [m/s]
Sn_left, Sp_left, Sn_right, Sp_right = 1e7, 0, 0, 1e7
sys.contact_S(Sn_left, Sp_left, Sn_right, Sp_right)

# Define a function for the generation rate
phi = 1e17 # photon flux [1/(cm^2 s)]
alpha = 2.3e4 # absorption coefficient [1/cm]
f = lambda x: phi * alpha * np.exp(-alpha * x)
sys.generation(f)

# First find the equilibrium solution
solution = sesame.solve_equilibrium(sys)
# IV curve
voltages = np.linspace(0, 0.95, 40)
j = sesame.IVcurve(sys, voltages, solution, '1d_homojunction_jv')

j = j * sys.scaling.current
result = {'v':voltages, 'j':j}

np.save('jv_values',result)
np.savetxt('jv_values.txt',(voltages,j))
try:
    import scipy.io.savemat as savemat
    savemat('jv_values.mat',result)
except ImportError:
    print("Scipy not installed, can't save to .mat file")


try:
    import matplotlib.pyplot as plt
    plt.plot(voltages,j,'-o')
    plt.xlabel('Voltage [V]')
    plt.ylabel('Current [A/cm^2]')
    plt.grid()
    plt.show()

except ImportError:
    print("Matplotlib not installed, can't make plot")



