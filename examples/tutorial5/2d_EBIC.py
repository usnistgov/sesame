import sesame
import numpy as np

######################################################
##      define the system
######################################################

# dimensions of the system
Lx = 3e-4   #[cm]
Ly = 3e-4   #[cm]

# extent of the junction from the left contact [cm]
junction = .1e-4    # [cm]

# Mesh
x = np.concatenate((np.linspace(0,.2e-4, 30, endpoint=False),
                    np.linspace(0.2e-4, 1.4e-4, 60, endpoint=False),
                    np.linspace(1.4e-4, 2.9e-4, 70, endpoint=False),
                    np.linspace(2.9e-4, 3e-4, 10)))

y = np.concatenate((np.linspace(0, 1.75e-4, 50, endpoint=False),
                    np.linspace(1.75e-4, 2.75e-4, 50, endpoint=False),
                    np.linspace(2.75e-4, Ly, 50)))

# Create a system
sys = sesame.Builder(x, y, periodic=False)

# Dictionary with the material parameters
mat = {'Nc':8e17, 'Nv':1.8e19, 'Eg':1.5, 'epsilon':9.4, 'Et': 0,
       'mu_e':320, 'mu_h':40, 'tau_e':10*1e-9, 'tau_h':10*1e-9, 'B': 1e-10}

# Add the material to the system
sys.add_material(mat)

# define a function specifiying the n-type region
def region(pos):
    x, y = pos
    return x < junction
# define a function specifiying the p-type region
region2 = lambda pos: 1 - region(pos)

# Add the donors
nD = 1e17 # [cm^-3]
sys.add_donor(nD, region)
# Add the acceptors
nA = 1e15 # [cm^-3]
sys.add_acceptor(nA, region2)

# Use Ohmic contacts
Sn_left, Sp_left, Sn_right, Sp_right = 1e7, 1e7, 1e7, 1e7
sys.contact_S(Sn_left, Sp_left, Sn_right, Sp_right)
sys.contact_type('Ohmic','Ohmic')

# gap state characteristics
E = 0                   # energy of gap state (eV) from midgap
rhoGB = 1e14            # density of defect states
s = 1e-14               # defect capture cross section

# this implies a surface recombination velocity S = rhoGB*s*vthermal = 1e5 [cm/s]

# Specify the two points that make the line containing additional recombination centers
p1 = (0, Ly)
p2 = (Lx, Ly)

# add neutral defect along surface (surface recombination boundary condition)
sys.add_line_defects([p1, p2], rhoGB, s, E=E, transition=(0,0))

# find equilibrium solution with GB.  Note we provide the GB-free equilibrium solution as a starting guess
solution = sesame.solve_equilibrium(sys, periodic_bcs=False)

######################################################
##      EBIC generation profile parameters
######################################################

q = 1.6e-19      # C
ibeam = 10e-12   # A
Ebeam = 15e3     # eV
eg = 1.5         # eV
density = 5.85   # g/cm^3
kev = 1e3        # eV
# rough approximation for total carrier generation rate from electron beam
Gtot = ibeam/q * Ebeam / (3*eg)
# length scale of generation volume
Rbulb = 0.043 / density * (Ebeam/kev)**1.75 # given in micron
Rbulb = Rbulb * 1e-4  # converting to cm
# Gaussian spread
sigma = Rbulb / np.sqrt(15)
# penetration depth
y0 = 0.3 * Rbulb
# get diffusion length to scale generation density
vt = .0258
Ld = np.sqrt(sys.mu_e[0] * sys.tau_e[0]) * sys.scaling.length
# converting Gtot to a 2-d quantity
Gtot = Gtot

######################################################
##      vary position of the electron beam
######################################################
x0list = np.linspace(.1e-4, 2.5e-4, 11)
# Array in which to store results
jset = np.zeros(len(x0list))
jratio = np.zeros(len(x0list))
rset = np.zeros(len(x0list))
rad_ratio = np.zeros(len(x0list))

# Cycle over beam positions
for idx, x0 in enumerate(x0list):

    # define a function for generation profile
    def excitation(x,y):
        return Gtot/(2*np.pi*sigma**2*Ld) * np.exp(-(x-x0)**2/(2*sigma**2)) * np.exp(-(y-Ly+y0)**2/(2*sigma**2))

    # add generation to the system at new beam position
    sys.generation(excitation)

    # solve the system
    solution = sesame.solve(sys, periodic_bcs=False, tol=1e-8)

    # get analyzer object to evaluate current and radiative recombination
    az = sesame.Analyzer(sys, solution)
    # compute (dimensionless) current and convert to to dimension-ful form
    tj = az.full_current() * sys.scaling.current * sys.scaling.length
    # save the current
    jset[idx] = tj
    # obtain total generation from sys object
    gtot = sys.gtot * sys.scaling.generation * sys.scaling.length**2
    jratio[idx] = tj/(q * gtot)

    # compute (dimensionless) total radiative recombination and convert to to dimension-ful form
    cl = az.integrated_radiative_recombination() * sys.scaling.generation * sys.scaling.length**2
    # save the CL
    rset[idx] = cl
    rad_ratio[idx] = cl/gtot

# display result
for counter in range(len(jset)):
    print('x = {0:2.1e} [cm], J = {1:5.3e} [mA/cm], CL = {2:5.3e}'.format(x0list[counter],jset[counter]*1e3,rset[counter]))

for counter in range(len(jset)):
    print('x = {0:2.1e} {1:5.3e} {2:5.3e}'.format(x0list[counter],jratio[counter],rad_ratio[counter]))

import matplotlib.pyplot as plt
plt.plot(x0list,jset)
plt.show()
