import sesame
import numpy as np

def system(amp=1):
    # Dimensions of the system
    Lx = 3e-6 # [m]
    Ly = 5e-6 # [m]
    # extent of the junction from the left contact [m]
    junction = 10e-9 

    # Mesh
    x = np.concatenate((np.linspace(0,1.2e-6, 100, endpoint=False), 
                        np.linspace(1.2e-6, Lx, 50)))
    y = np.concatenate((np.linspace(0, 2.25e-6, 50, endpoint=False), 
                        np.linspace(2.25e-6, 2.75e-6, 50, endpoint=False),
                        np.linspace(2.75e-6, Ly, 50)))

    sys = sesame.Builder(x, y)

    # Add the donors
    nD = 1e17 * 1e6 # [m^-3]
    sys.add_donor(nD, lambda pos: pos[0] < junction)

    # Add the acceptors
    nA = 1e15 * 1e6 # [m^-3]
    sys.add_acceptor(nA, lambda pos: pos[0] >= junction)

    # Use perfectly selective contacts
    Sn_left, Sp_left, Sn_right, Sp_right = 1e50, 0, 0, 1e50
    sys.contacts(Sn_left, Sp_left, Sn_right, Sp_right)

    # Region 1
    reg1 = {'Nc':8e17*1e6, 'Nv':1.8e19*1e6, 'Eg':1.5, 'epsilon':9.4,
            'mu_e':100*1e-4, 'mu_h':100*1e-4, 'tau_e':10e-9, 'tau_h':10e-9, 
            'Et':0, 'band_offset':0, 'B':0, 'Cn':0, 'Cp':0}
    sys.add_material(reg1, lambda pos: (pos[1] <= 2.4e-6) | (pos[1] >= 2.6e-6))

    # Region 2
    reg2 = {'Nc':8e17*1e6, 'Nv':1.8e19*1e6, 'Eg':1.5, 'epsilon':9.4,
            'mu_e':20*1e-4, 'mu_h':20*1e-4, 'tau_e':10e-9, 'tau_h':10e-9, 
            'Et':0, 'band_offset':0, 'B':0, 'Cn':0, 'Cp':0}
    sys.add_material(reg2, lambda pos: (pos[1] > 2.4e-6) & (pos[1] < 2.6e-6))

    # gap state characteristics
    S = 1e5 * 1e-2           # trap recombination velocity [m/s]
    E = -0.25                # energy of gap state (eV) from midgap
    N = amp * 2e14 * 1e4           # defect density [1/m^2]

    p1 = (20e-9, 2.5e-6)   #[m]
    p2 = (2.9e-6, 2.5e-6)  #[m]

    sys.add_line_defects([p1, p2], E, N, S)

    # Define a function for the generation rate
    phi = 1e21 # photon flux [1/(m^2 s)]
    alpha = 2.3e6 # absorption coefficient [1/m]
    f = lambda x, y: phi * alpha * np.exp(-alpha * x)
    sys.generation(f)

    return sys



if __name__ == '__main__':

    #---------------------------------------------------------------
    #                           Step 1
    #---------------------------------------------------------------
    # Create the system with reduced defect density of states
    sys = system(0.0001)

    # Electrostatic potential
    v_left  = np.log(sys.rho[0]/sys.Nc[0])
    v_right = -sys.Eg[sys.nx-1] - np.log(-sys.rho[sys.nx-1]/sys.Nv[sys.nx-1])
    v = np.linspace(v_left, v_right, sys.nx)
    v = np.tile(v, sys.ny) # replicate the guess in the y-direction

    # Call Poisson solver
    v = sesame.poisson_solver(sys, v)

    #---------------------------------------------------------------
    #                           Step 2
    #---------------------------------------------------------------
    # Initial arrays for the quasi-Fermi levels
    efn = np.zeros((sys.nx*sys.ny,))
    efp = np.zeros((sys.nx*sys.ny,))
    result = {'efn':efn, 'efp':efp, 'v':v}

    # Loop at zero bias with increasing defect density of states
    for amp in [0.0001, 0.01]:
        sys = system(amp)
        result = sesame.ddp_solver(sys, result, eps=1)

    #---------------------------------------------------------------
    #                           Step 3
    #---------------------------------------------------------------
    # Create the system with the defect density of states we want
    sys = system()

    # Loop over the applied potentials
    voltages = np.linspace(0, 1, 40)
    sesame.IVcurve(sys, voltages, result, '2dpnIV.vapp', eps=1)
