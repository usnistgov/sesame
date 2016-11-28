import sesame
import numpy as np

def system():
    # Dimensions of the system
    Lx = 3e-6 # [m]
    Ly = 5e-6 # [m]
    # extent of the junction from the left contact [m]
    junction = 10e-9 

    # Mesh
    x = np.concatenate((np.linspace(0,1.2e-6, 300, endpoint=False), 
                        np.linspace(1.2e-6, Lx, 100)))
    y = np.concatenate((np.linspace(0, 2.25e-6, 100, endpoint=False), 
                        np.linspace(2.25e-6, 2.75e-6, 100, endpoint=False),
                        np.linspace(2.75e-6, Ly, 100)))

    sys = sesame.Builder()

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
            'mu_e':200*1e-4, 'mu_h':200*1e-4, 'tau_e':10e-9, 'tau_h':10e-9, 
            'RCenergy':0, 'band_offset':0}
    sys.add_material(reg1, lambda pos: pos[1] <= 2.4e-6 or pos[1] >= 2.6e-6)

    # Region 2
    reg2 = {'Nc':8e17*1e6, 'Nv':1.8e19*1e6, 'Eg':1.5, 'epsilon':9.4,
            'mu_e':20*1e-4, 'mu_h':20*1e-4, 'tau_e':10e-9, 'tau_h':10e-9, 
            'RCenergy':0, 'band_offset':0}
    sys.add_material(reg2, lambda pos: pos[1] > 2.4e-6 and pos[1] < 2.6e-6)

    # gap state characteristics
    S = 1e5 * 1e-2           # trap recombination velocity [m/s]
    E = -0.25                # energy of gap state (eV) from midgap
    N = 2e14 * 1e4           # defect density [1/m^2]

    p1 = (20e-9, 2.5e-6)   #[m]
    p2 = (2.9e-6, 2.5e-6)  #[m]

    sys.add_line_defects([p1, p2], E, N, S)

    sys.finalize()
    return sys



if __name__ == '__main__':
    sys = system()
    v_left  = np.log(abs(sys.rho[0])/sys.Nc[0])
    v_right = -sys.Eg[0] - np.log(abs(sys.rho[sys.nx-1])/sys.Nv[sys.nx-1])

    # Initial guess
    v = np.empty((sys.nx,), dtype=float) 
    v[:sys.nx] = np.linspace(v_left, v_right, sys.nx)
    v = np.tile(v, sys.ny) # replicate the guess in the y-direction

    # Call Poisson solver with a tolerance of 10^-9
    v = sesame.poisson_solver(sys, v, 1e-9, info=1, max_step=100)

    # Initial arrays for the quasi-Fermi levels
    efn = np.zeros((sys.nx*sys.ny,))
    efp = np.zeros((sys.nx*sys.ny,))

    # Loop over the applied potentials made dimensionless
    applied_voltages = np.linspace(0, 1, 41) / sys.vt
    for idx, vapp in enumerate(applied_voltages):
        # Apply the contacts boundary conditions
        for i in range(0, sys.nx*(sys.ny-1)+1, sys.nx):
            v[i] = v_left
            v[i+sys.nx-1] = v_right + vapp

        # Call the Drift Diffusion Poisson solver with tolerance 10^-9
        result = sesame.ddp_solver(sys, (efn, efp, v), 1e-9, max_step=30, info=1)
        if result is not None:
            # Extract the results from the dictionary 'result'
            v = result['v']
            efn = result['efn']
            efp = result['efp']

            # Save the data
            np.save("data.vapp_idx_{0}".format(idx), [efn, efp, v])
