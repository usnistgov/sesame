import sesame
import numpy as np
import scipy.io

def runTest7():

    L = 4e-4 # length of the system in the x-direction [m]
    dd = .05e-4
    Ly = 2e-4

    # Mesh
    x = np.concatenate((np.linspace(0,1e-4-dd, 70, endpoint=False),
                        np.linspace(1e-4-dd, 1e-4+dd, 20, endpoint=False),
                        np.linspace(1e-4+dd,3e-4-dd,140, endpoint=False),
                        np.linspace(3e-4-dd,3e-4+dd,20, endpoint=False),
                        np.linspace(3e-4+dd,L,70)))
    y = np.concatenate((np.linspace(0,1e-4-dd,70, endpoint=False),
                        np.linspace(1e-4-dd,1e-4+dd,20, endpoint=False),
                        np.linspace(1e-4+dd,2e-4,70 )))

    # Create a system
    sys = sesame.Builder(x,y, periodic=False)

    tau = 1e-8
    vt = 0.025851991024560

    Nc1 = 1e17
    Nv1 = 1e18

    Nc2 = 1e17
    Nv2 = 1e18

    # Dictionary with the material parameters
    mat1 = {'Nc':Nc1, 'Nv':Nv1, 'Eg':1., 'epsilon':10, 'Et': 0,
            'mu_e':100, 'mu_h':40, 'tau_e':tau, 'tau_h':tau,
            'affinity': 4.15}

    mat2 = {'Nc':Nc2, 'Nv':Nv2, 'Eg':1.1, 'epsilon':100, 'Et': 0,
            'mu_e':100, 'mu_h':40, 'tau_e':tau, 'tau_h':tau,
            'affinity': 4.05}

    junction = 2e-4 # extent of the junction from the left contact [m]

    def region1(pos):
        x, y = pos
        val = x <= 1e-4
        return val
    def region2(pos):
        x, y = pos
        val = (x> 1e-4) & (x < 3e-4) & (y >= 1e-4)
        return val
    def region3(pos):
        x, y = pos
        val = (x > 1e-4) & (x < 3e-4) & (y < 1e-4)
        return val
    def region4(pos):
        x, y = pos
        val = x >= 3e-4
        return val

    # Add the material to the system
    sys.add_material(mat1, region1)
    sys.add_material(mat1, region2)
    sys.add_material(mat2, region3)
    sys.add_material(mat2, region4)

    # Add the donors
    nD1 = 1e15 # [cm^-3]
    sys.add_donor(nD1, region1)
    sys.add_donor(nD1, region2)
    nD2 = 1e15 # [cm^-3]
    sys.add_acceptor(nD2, region3)
    sys.add_acceptor(nD2, region4)

    # Define the surface recombination velocities for electrons and holes [m/s]
    sys.contact_type('Ohmic','Ohmic')
    SS = 1e50
    Sn_left, Sp_left, Sn_right, Sp_right = SS, SS, SS, SS
    sys.contact_S(Sn_left, Sp_left, Sn_right, Sp_right)

    # Electrostatic potential dimensionless

    solution = sesame.solve(sys, compute='Poisson', verbose=False)
    veq = np.copy(solution['v'])

    solution.update({'x': sys.xpts, 'y': sys.ypts, 'chi': sys.bl, 'eg': sys.Eg, 'Nc': sys.Nc, 'Nv': sys.Nv, 'epsilon': sys.epsilon})


    # IV curve

    solution.update({'efn': np.zeros((sys.nx*sys.ny,)), 'efp': np.zeros((sys.nx*sys.ny,))})

    G = 1*1e18
    f = lambda x, y: G

    sys.generation(f)
    solution = sesame.solve(sys, guess=solution, maxiter=5000, verbose=False)
    az = sesame.Analyzer(sys, solution)
    tj = -az.full_current()

    voltages = np.linspace(.0, .8, 9)

    result = solution

    # sites of the right contact
    nx = sys.nx
    s = [nx - 1 + j * nx for j in range(sys.ny)]

    # sign of the voltage to apply
    if sys.rho[nx - 1] < 0:
        q = 1
    else:
        q = -1

    j = []
    # Loop over the applied potentials made dimensionless
    Vapp = voltages / sys.scaling.energy
    for idx, vapp in enumerate(Vapp):

        # Apply the voltage on the right contact
        result['v'][s] = veq[s] + q * vapp
        # Call the Drift Diffusion Poisson solver
        result = sesame.solve(sys, guess=result, maxiter=1000, verbose=False)
        # Compute current
        az = sesame.Analyzer(sys, result)
        tj = az.full_current() * sys.scaling.current * sys.scaling.length / (2e-4)
        j.append(tj)



    jcomsol = np.array([0.50272, 0.48515, 0.40623, -0.16696, -5.1204, -58.859, -819.11, -7024.4, -27657])
    jcomsol = jcomsol * 1e-4
    error = np.max(np.abs((jcomsol-np.transpose(j))/(.5*(jcomsol+np.transpose(j)))))
    print("error = {0}".format(error))

