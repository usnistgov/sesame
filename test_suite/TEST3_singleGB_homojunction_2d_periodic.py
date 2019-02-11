import sesame
import numpy as np
import scipy.io as sio

def system(N=0,s=1e-18*1e4):
    # dimensions of the system
    Lx = 3e-6*1e2 #[m]
    Ly = 3e-6*1e2 #[m]

    # extent of the junction from the left contact [m]
    junction = .1e-6*1e2

    ## initial: 60,50,10...  40,20,40
    # Mesh
    x = np.concatenate((np.linspace(0,.2e-6*1e2, 30, endpoint=False),
                        np.linspace(0.2e-6*1e2, 1.4e-6*1e2, 60, endpoint=False),
                        np.linspace(1.4e-6*1e2, 2.7e-6*1e2, 60, endpoint=False),
                        np.linspace(2.7e-6*1e2, Lx-0.02e-6*1e2, 30, endpoint=False),
                        np.linspace(Lx-0.02e-6*1e2, Lx, 10)))

    y = np.concatenate((np.linspace(0, 1.25e-6*1e2, 60, endpoint=False),
                        np.linspace(1.25e-6*1e2, 1.75e-6*1e2, 50, endpoint=False),
                        np.linspace(1.75e-6*1e2, Ly, 60)))


    # Create a system
    sys = sesame.Builder(x, y)

    def region(pos):
        x, y = pos
        return x < junction

    # Add the donors
    nD = 1e17 # [m^-3]
    sys.add_donor(nD, region)

    # Add the acceptors
    region2 = lambda pos: 1 - region(pos)
    nA = 1e15 # [m^-3]
    sys.add_acceptor(nA, region2)

    # Define Ohmic contacts
    sys.contact_type('Ohmic', 'Ohmic')

    # Use perfectly selective contacts
    Sn_left, Sp_left, Sn_right, Sp_right = 1e50, 1e50, 1e50, 1e50
    sys.contact_S(Sn_left, Sp_left, Sn_right, Sp_right)

    Nc = 8e17
    Nv = 1.8e19
    q = 1.60217662*1e-19
    kb = 1.38064852*1e-23
    t = 300
    vt = kb*t/q

    # Dictionary with the material parameters
    mat = {'Nc':Nc, 'Nv':Nv, 'Eg':1.5, 'epsilon':9.4, 'Et': 0*vt*np.log(Nc/Nv),
            'mu_e':320, 'mu_h':40, 'tau_e':10*1e-9, 'tau_h':10*1e-9}

    # Add the material to the system
    sys.add_material(mat)


    # gap state characteristics
    E = 0.4 + .5*vt*np.log(Nc/Nv)                # energy of gap state (eV) from midgap

    # Specify the two points that make the line containing additional charges
    p1 = (.1e-6*1e2, 1.5*1e-6*1e2)   #[m]
    p2 = (2.9e-6*1e2, 1.5*1e-6*1e2)  #[m]

    # Pass the information to the system
    sys.add_defects([p1, p2], N, s, E=E, transition=(1,0))
    sys.add_defects([p1, p2], N, s, E=E, transition=(0,-1))

    return sys

def runTest3():

    rhoGBlist = np.linspace(1e6*1e-4,1e18*1e-4,2)

    sys = system(rhoGBlist[0])

    solution = sesame.solve(sys, compute='Poisson', verbose=False)



    s0 = 1e-18*1e4
    rhoGBlist = [1e6*1e-4, 1e18*1e-4]
    for idx, rhoGB in enumerate(rhoGBlist):
        sys = system(rhoGB,s0)
        solution = sesame.solve(sys, compute='Poisson', guess=solution, maxiter=5000, verbose=False)
    veq = np.copy(solution['v'])

    efn = np.zeros((sys.nx * sys.ny,))
    efp = np.zeros((sys.nx * sys.ny,))
    solution.update({'efn': efn, 'efp': efp})

    junction = .1e-6*1e2
    # Define a function for the generation rate

    G = 1
    phi0 = 1e21 * G * 1e-4
    alpha = 2.3e6 * 1e-2  # alpha = 2e4 cm^-1 for CdTe
    f = lambda x, y: phi0 * alpha * np.exp(-alpha * x)
    sys.generation(f)

    slist = [1e-18 * 1e4]

    sys = system(rhoGBlist[1],slist[0])

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
        tj = az.full_current() * sys.scaling.current * sys.scaling.length / (3e-6*1e2)
        j.append(tj)
    #    print(j)


    jSesame_12_4_2017 = np.array([135.14066065175203, 134.97430561196626, 134.70499402818209, 134.28271667573679, 133.27884008619145, 129.49875552490002, 119.14704988797484, 83.157765739151415, -114.57979137988193])
    jSesame_12_4_2017 = jSesame_12_4_2017 * 1e-4
    error = np.max(np.abs((jSesame_12_4_2017-np.transpose(j))/(.5*(jSesame_12_4_2017+np.transpose(j)))))
    print("error = {0}".format(error))

