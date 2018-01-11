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
                    np.linspace(1.2e-6, 2.9e-6, 50, endpoint=False),
                    np.linspace(2.9e-6, Lx, 10)))
y = np.linspace(0, Ly, 100)
z = np.linspace(0, Lz, 50)

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

# Define Ohmic contacts
sys.contact_type('Ohmic', 'Ohmic')

# Use perfectly selective contacts
Sn_left, Sp_left, Sn_right, Sp_right = 1e50, 0, 0, 1e50
sys.contact_S(Sn_left, Sp_left, Sn_right, Sp_right)

# dictionary with the material parameters
CdTe = {'Nc':8e17*1e6, 'Nv':1.8e19*1e6, 'Eg':1.5, 'epsilon':9.4,
        'mu_e':100*1e-4, 'mu_h':100*1e-4, 'tau_e':10e-9, 'tau_h':10e-9}

# add the material to the system
sys.add_material(CdTe)

# gap state characteristics
s = 1e-15 * 1e-4         # trap capture cross section [m^2]
E = -0.25                # energy of gap state (ev) from midgap
N = 2e13 * 1e4           # defect density [1/m^2]

# specify the four points that define the plane containing additional charges
p1 = (1e-6, .5e-6, 1e-6)    #[m]
p2 = (2.9e-6, .5e-6, 1e-6)  #[m]

q1 = (1.0e-6, 4.5e-6, 1e-9) #[m]
q2 = (2.9e-6, 4.5e-6, 1e-9) #[m]

# pass the information to the system
sys.add_plane_defects([p1, p2, q1, q2], N, s, E=E)

# visualize the system
sesame.plot_plane_defects(sys)
