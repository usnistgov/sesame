import sesame
import numpy as np
import scipy.io

def runTest6():

    L = 4e-6*1e2 # length of the system in the x-direction [m]
    Ly = 2e-6*1e2
    dd = .005e-6*1e2

    # Mesh
    x = np.concatenate((np.linspace(0,L/2-dd, 100, endpoint=False),
                        np.linspace(L/2-dd, L/2+dd, 20, endpoint=False),
                        np.linspace(L/2+dd,L, 100)))

    y = np.linspace(0,Ly,30)

    # Create a system
    sys = sesame.Builder(x,y)

    tau = 1e8
    vt = 0.025851991024560

    Nc1 = 2.2*1e18
    Nv1 = 2.2*1e18

    Nc2 = 2.2*1e18
    Nv2 = 2.2*1e18

    # Dictionary with the material parameters
    mat1 = {'Nc':Nc1, 'Nv':Nv1, 'Eg':1.5, 'epsilon':1000, 'Et': 0,
            'mu_e':100, 'mu_h':40, 'tau_e':tau, 'tau_h':tau,
            'affinity': 4.05}

    mat2 = {'Nc':Nc2, 'Nv':Nv2, 'Eg':1.5, 'epsilon':10000, 'Et': 0,
            'mu_e':100, 'mu_h':40, 'tau_e':tau, 'tau_h':tau,
            'affinity': 4.05}

    junction = 2e-6*1e2 # extent of the junction from the left contact [m]

    def region1(pos):
        x, y = pos
        return x < junction

    # Add the acceptors
    region2 = lambda pos: 1 - region1(pos)

    # Add the material to the system
    sys.add_material(mat1, region1)
    sys.add_material(mat2, region2)

    # Add the donors
    nD1 = 1e15 # [m^-3]
    sys.add_donor(nD1, region1)
    nD2 = 1e15 # [m^-3]
    sys.add_acceptor(nD2, region2)

    # Define the surface recombination velocities for electrons and holes [m/s]
    sys.contact_type('Ohmic','Ohmic')
    SS = 1e50
    Sn_left, Sp_left, Sn_right, Sp_right = SS, SS, SS, SS
    sys.contact_S(Sn_left, Sp_left, Sn_right, Sp_right)

    # Electrostatic potential dimensionless

    solution = sesame.solve(sys, compute='Poisson', periodic_bcs=False, verbose=False)
    veq = np.copy(solution['v'])

    solution.update({'x': sys.xpts, 'chi': sys.bl, 'eg': sys.Eg, 'Nc': sys.Nc, 'Nv': sys.Nv, 'epsilon': sys.epsilon})

    # IV curve
    solution.update({'efn': np.zeros((sys.nx*sys.ny,)), 'efp': np.zeros((sys.nx*sys.ny,))})

    G = 1*1e24 * 1e-6
    f = lambda x, y: G
    sys.generation(f)

    solution = sesame.solve(sys, guess=solution, verbose=False)
    solution.update({'x': sys.xpts, 'chi': sys.bl, 'eg': sys.Eg, 'Nc': sys.Nc, 'Nv': sys.Nv})

    voltages = np.linspace(0, 0.9, 10)

    result = solution

    # sites of the right contact
    nx = sys.nx
    s = [nx-1 + j*nx for j in range(sys.ny)]

    # sign of the voltage to apply
    if sys.rho[nx-1] < 0:
        q = 1
    else:
        q = -1

    j = []
    # Loop over the applied potentials made dimensionless
    Vapp = voltages / sys.scaling.energy
    for idx, vapp in enumerate(Vapp):

        # Apply the voltage on the right contact
        result['v'][s] = veq[s] + q*vapp
        # Call the Drift Diffusion Poisson solver
        result = sesame.solve(sys, guess=result, maxiter=1000, verbose=False)
        # Compute current
        az = sesame.Analyzer(sys, result)
        tj = az.full_current()* sys.scaling.current * sys.scaling.length / (Ly)
        j.append(tj)


    jcomsol = np.array([0.55569,0.54937,0.5423,0.53436,0.52535,0.51499,0.50217,0.4622,-0.47448,-31.281])
    jcomsol = jcomsol * 1e-4
    error = np.max(np.abs((jcomsol-np.transpose(j))/(.5*(jcomsol+np.transpose(j)))))

    print("error = {0}".format(error))

