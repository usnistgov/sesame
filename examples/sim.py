import sesame
import numpy as np

#==============================================================================
#       System
#==============================================================================
sys = sesame.Builder()

# dimensions of the system
L = 3e-6 # [m]
d = 5e-6 # [m]
junction = 10e-9 # size of the n-region [m]

# dictionary with the material parameters
CdTe = {'Nc':8e17*1e6, 'Nv':1.8e19*1e6, 'Eg':1.5, 'epsilon':9.4,
        'mu_e':100*1e-4, 'mu_h':100*1e-4, 'tau_e':10e-9, 'tau_h':10e-9, 'RCenergy':0}

# add the material to the system
sys.add_material(((0,0,0), (L,d,0)), CdTe)

# add the donors
nD = 1e17 * 1e6 # [m^-3]
sys.add_donor(((0,0,0), (junction,d,0)), nD)

# add the acceptors
nA = 1e15 * 1e6 # [m^-3]
sys.add_acceptor(((junction,0,0), (L,d,0)), nA)

# define the surface recombination velocities for electrons and holes [m/s]
Sn_left, Sp_left, Sn_right, Sp_right = 1e50, 0, 0, 1e50
sys.contacts(Sn_left, Sp_left, Sn_right, Sp_right)

# mesh
x = np.concatenate((np.linspace(0,1.2e-6, 300, endpoint=False), 
                    np.linspace(1.2e-6, L, 100)))
y = np.concatenate((np.linspace(0, 2.25e-6, 100, endpoint=False), 
                    np.linspace(2.25e-6, 2.75e-6, 100, endpoint=False),
                    np.linspace(2.75e-6, d, 100)))
sys.mesh(x, y)

#generation profile::
phi = 1e21 # photon flux [1/(m^2 s)]
alpha = 2.3e6 # absorption coefficient [1/m]

# define a function for the generation rate
f = lambda x, y, z: phi * alpha * np.exp(-alpha * x)
sys.illumination(f)

# defect line
S   = 1e5 * 1e-2           # trap recombination velocity [m/s]
EGB = -0.25                # energy of gap state (eV) from midgap
NGB = 2e14 * 1e4           # defect density. [1/m^2]

# specify the start and end point of the line containing additional charges
startGB = (20e-9, 2.5e-6, 0)   #[m]
endGB   = (2.8e-6, 2.5e-6, 0)  #[m]

# pass the information to the system
sys.add_local_charges([startGB, endGB], EGB, NGB, S)

# finalyze
sys.finalize()


#==============================================================================
#       Calculation
#==============================================================================

if __name__ == '__main__':
    # prepare the guess for the electrostatic potential
    v00 = np.log(abs(sys.rho[0])/sys.Nc[0])
    vL0 = -sys.Eg[0] - np.log(abs(sys.rho[sys.nx-1])/sys.Nv[sys.nx-1])

    v = np.empty((sys.nx,), dtype=float)
    v[:sys.nx] = np.linspace(v00, vL0, sys.nx)
    v = np.tile(v, sys.ny)

    v = sesame.poisson_solver(sys, v, 1e-9, info=1, max_step=1000)

    efn = np.zeros((sys.nx*sys.ny,))
    efp = np.zeros((sys.nx*sys.ny,))
    for vapp in np.linspace(0, 40, 41):
        nx, ny = sys.nx, sys.ny
        for i in range(0, nx*(ny-1)+1, nx):
            v[i] = v00
            v[i+nx-1] = vL0 + vapp

        result = sesame.ddp_solver(sys, (efn, efp, v), 1e-9, max_step=30, info=1)
        
        if result is None:
            print("no result for vapp = {0}".format(vapp))
            exit(1)
        
        if result is not None:
            v = result['v']
            efn = result['efn']
            efp = result['efp']

            np.save("data.vapp_{0}".format(vapp), [efn, efp, v])
