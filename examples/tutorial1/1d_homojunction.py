import sesame
import numpy as np

L = 3e-4 # length of the system in the x-direction [cm]

# Mesh
x = np.concatenate((np.linspace(0,1.2e-4, 100, endpoint=False),
                    np.linspace(1.2e-4, L, 50)))

# Create a system
sys = sesame.Builder(x)

# Dictionary with the material parameters
material = {'Nc':8e17, 'Nv':1.8e19, 'Eg':1.5, 'affinity':3.9, 'epsilon':9.4,
        'mu_e':100, 'mu_h':100, 'tau_e':10e-9, 'tau_h':10e-9, 'Et':0}

# Add the material to the system
sys.add_material(material)

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

# Define the surface recombination velocities for electrons and holes [cm/s]
Sn_left, Sp_left, Sn_right, Sp_right = 1e7, 0, 0, 1e7
sys.contact_S(Sn_left, Sp_left, Sn_right, Sp_right)

# Define a function for the generation rate
phi = 1e17         # photon flux [1/(cm^2 s)]
alpha = 2.3e4      # absorption coefficient [1/cm]

# Define a function for the generation rate
def gfcn(x,y):
    return phi * alpha * np.exp(-alpha * x)

# add generation to system
sys.generation(gfcn)

# IV curve
voltages = np.linspace(0, 0.95, 10)
j = sesame.IVcurve(sys, voltages, '1dhomo_V')

# convert dimensionless current to dimension-ful current
j = j * sys.scaling.current
# save voltage and current values to dictionary
result = {'v':voltages, 'j':j}

# save data to python data file
np.save('IV_values', result)

# save data to an ascii txt file
np.savetxt('IV_values.txt', (voltages, j))

# save data to a matlab data file
try:
    from scipy.io import savemat
    savemat('IV_values.mat', result)
# no scipy installed
except ImportError:
    print("Scipy not installed, can't save to .mat file")

# plot I-V curve
try:
    import matplotlib.pyplot as plt
    plt.plot(voltages, j,'-o')
    plt.xlabel('Voltage [V]')
    plt.ylabel('Current [A/cm^2]')
    plt.grid()     # add grid
    plt.show()     # show the plot on the screen
# no matplotlib installed
except ImportError:
    print("Matplotlib not installed, can't make plot")

