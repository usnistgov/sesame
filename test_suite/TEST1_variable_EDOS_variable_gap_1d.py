import sesame
import numpy as np
import scipy.io

def runTest1():

    L = 3e-4 # length of the system in the x-direction [cm]

    dd = 1e-7*1e2
    # Mesh
    x = np.concatenate((np.linspace(0,5e-7*1e2-dd,50, endpoint=False),
                        np.linspace(5e-7*1e2-dd,5e-7*1e2+dd,20, endpoint=False),
                        np.linspace(5e-7*1e2+dd,1e-6*1e2, 100, endpoint=False),
                        np.linspace(1e-6*1e2, L, 200)))

    # Create a system
    sys = sesame.Builder(x,input_length='cm')

    tau = 1e-8
    vt = 0.025851991024560

    Nc1 = 8*1e17
    Nv1 = 1.8*1e19

    Nc2 = 8*1e17
    Nv2 = 1.8*1e19

    # Dictionary with the material parameters
    mat1 = {'Nc':Nc1, 'Nv':Nv1, 'Eg':2.4, 'epsilon':10, 'Et': 0,
            'mu_e':320, 'mu_h':40, 'tau_e':tau, 'tau_h':tau,
            'affinity': 4.15}

    mat2 = {'Nc':Nc2, 'Nv':Nv2, 'Eg':1.5, 'epsilon':10, 'Et': 0,
            'mu_e':320, 'mu_h':40, 'tau_e':tau, 'tau_h':tau,
            'affinity': 4.05}

    junction = 5e-7*1e2 # extent of the junction from the left contact [m]
    def region1(pos):
        x = pos
        return x < junction

    # Add the acceptors
    region2 = lambda pos: 1 - region1(pos)

    # Add the material to the system
    sys.add_material(mat1, region1)
    sys.add_material(mat2, region2)

    # Add the donors
    nD1 = 1e17 # [cm^-3]
    sys.add_donor(nD1, region1)
    nD2 = 1e15 # [cm^-3]
    sys.add_acceptor(nD2, region2)

    # Define Ohmic contacts
    sys.contact_type('Ohmic', 'Ohmic')

    # Define the surface recombination velocities for electrons and holes [m/s]
    SS = 1e50*1e2
    Sn_left, Sp_left, Sn_right, Sp_right = SS, SS, SS, SS
    sys.contact_S(Sn_left, Sp_left, Sn_right, Sp_right)

    solution = sesame.solve(sys, compute='Poisson', verbose=False)
    veq = np.copy(solution['v'])

    G = 1*1e24*1e-6
    f = lambda x: G
    sys.generation(f)

    solution = sesame.solve(sys, guess=solution, verbose=False)
    solution.update({'x': sys.xpts, 'chi': sys.bl, 'eg': sys.Eg, 'Nc': sys.Nc, 'Nv': sys.Nv})

    voltages = np.linspace(0, 0.8, 9)

    result = solution

    # sites of the right contact
    nx = sys.nx
    s = [nx-1 + j*nx  for j in range(sys.ny)]

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
        tj = az.full_current()* sys.scaling.current
        j.append(tj)



    jcomsol = np.array([0.32117,0.31672,0.31198,0.30683,0.30031,0.28562,0.20949,-0.39374,-7.0681])
    jcomsol = jcomsol * 1e-4 # converting to A/cm^2

    error = np.max(np.abs((jcomsol-np.transpose(j))/(.5*(jcomsol+np.transpose(j)))))
    print("error = {0}".format(error))

