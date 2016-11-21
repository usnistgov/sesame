Make a system and solve the problem
===================================

Build a system
--------------

Suppose we want to simulate a pn junction as depicted below.  We start by
importing the sesame package and numpy::

    import sesame
    import numpy as np

Create an instance of the builder::

    sys = sesame.Builder()

and add features to it.  Define the material using a dictionary and add it to
the system::

    CdTe = {'Nc':8e17*1e6, 'Nv':1.8e19*1e6, 'Eg':1.5, 'epsilon':9.4,
            'mu_e':100*1e-4, 'mu_h':100*1e-4, 'tau_e':10e-9, 'tau_h':10e-9, 'RCenergy':0}

    sys.add_material(((0,0,0), (Lx,Ly,Lz)), CdTe)

where Nc (Nv) is ... Add doping concentrations::

    sys.add_donor(((0,0,0), (junction,Ly,Lz)), nD)
    sys.add_acceptor(((junction,0,0), (Lx,Ly,Lz)), nA)

The contacts boundary conditions::

    sys.contacts(1e48, 0, 0, 1e48)

Create a mesh and add it to the system::

    x = np.concatenate((np.linspace(0,1.2e-6, 300, endpoint=False), 
                        np.linspace(1.2e-6, Lx, 100)))
    y = np.concatenate((np.linspace(0, 2.25e-6, 100, endpoint=False), 
                        np.linspace(2.25e-6, 2.75e-6, 100, endpoint=False),
                        np.linspace(2.75e-6, Ly, 100)))
    sys.mesh(x, y)

Add a generation profile::

    phi = 1e27
    alpha = 2e6 # alpha = 2e4 cm^-1 for CdTe
    f = lambda x, y, z: phi * np.exp(-alpha * x)
    sys.illumination(f)

Add local charges, to simulate a grain boundary for example::

    S   = 1e5 * 1e-2           # trap recombination velocity (m/s)
    EGB = -0.25                # energy of gap state (eV) from midgap
    NGB = 2e14 * 1e4           # defect density. (1/m^2)

    # specify the start and end point of the line containing additional charges
    startGB = (20e-9, 2.5e-6, 0)
    endGB = (2.8e-6, 2.5e-6, 0)

    sys.add_local_charges([startGB, endGB], EGB, NGB, S)

Finalyze the system::

    sys.finalyze()

Run calculations and save data
------------------------------

A good way to start is by computing the thermal equilibrium electrostatic
potential::

    # Left and right potentials
    v_left  = np.log(abs(sys.rho[0])/sys.Nc[0])
    v_right = -sys.Eg[0] - np.log(abs(sys.rho[sys.nx-1])/sys.Nv[sys.nx-1])

    # Initial guess
    v = np.empty((sys.nx,), dtype=float)
    v[:sys.nx] = np.linspace(v_left, v_right, sys.nx)
    v = np.tile(v, sys.ny)

    # Call Poisson solver
    v = sesame.poisson_solver(sys, v, 1e-9, info=1, max_step=1000)

Then we can solve the drift difussion Poisson equations to compute a
J(V) characteristics. The call to the drift diffusion Poisson solver returns a
dictionary with all values of electrostatic potnetial and quasi-Fermi levels. In
the following we solve the problem for multiple applied voltages and save the
output after each step::

    # Initial arrays for the quasi-Fermi levels
    efn = np.zeros((sys.nx*sys.ny,))
    efp = np.zeros((sys.nx*sys.ny,))

    # Loop over the applied potentials
    for vapp in np.linspace(0, 40, 41):
        # Apply the contacts boundary conditions
        for i in range(0, sys.nx*(sys.ny-1)+1, sys.nx):
            v[i] = v_right
            v[i+sys.nx-1] = v_left + vapp

        # Call the Drift Diffusion Poisson solver
        result = sesame.ddp_solver(sys, (efn, efp, v), 1e-9, max_step=30, info=1)
        
        if result is None:
            print("no result for vapp = vapp)
            exit(1)
        
        if result is not None:
            # Extract the results from the dictionary 'result'
            v = result['v']
            efn = result['efn']
            efp = result['efp']

            # Save the data
            np.save("data.vapp_{0}".format(vapp), [efn, efp, v])
