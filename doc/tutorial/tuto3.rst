Tutorial 3: Computing a J(V) characteristics
-----------------------------------------------
In this tutorial we show how to solve the Poisson equation at thermal
equilibrium and the drift diffusion Poisson equations under nonequilibrium
conditions.

.. seealso:: The example treated here is in the file ``sim.py`` in the
   ``examples`` directory in the root directory of the distribution. 


We consider the two-dimensional system created in :doc:`tutorial 2 <tuto2>` that
we rewrite inside its own function::

    import sesame
    import numpy as np

    def system():
        sys = sesame.Builder()
        
        # Dimensions of the system
        Lx = 3e-6 # [m]
        Ly = 5e-6 # [m]
        # extent of the junction from the left contact [m]
        junction = 10e-9 

        # Add the donors
        nD = 1e17 * 1e6 # [m^-3]
        sys.add_donor(nD, lambda pos: pos[0] < junction)

        # Add the acceptors
        nA = 1e15 * 1e6 # [m^-3]
        sys.add_acceptor(nA, lambda pos: pos[0] >= junction)

        # Use perfectly selective contacts
        Sn_left, Sp_left, Sn_right, Sp_right = 1e50, 0, 0, 1e50
        sys.contacts(Sn_left, Sp_left, Sn_right, Sp_right)


        # Mesh
        x = np.concatenate((np.linspace(0,1.2e-6, 300, endpoint=False), 
                            np.linspace(1.2e-6, Lx, 100)))
        y = np.concatenate((np.linspace(0, 2.25e-6, 100, endpoint=False), 
                            np.linspace(2.25e-6, 2.75e-6, 100, endpoint=False),
                            np.linspace(2.75e-6, Ly, 100)))
        sys.mesh(x, y)

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

        p1 = (20e-9, 2.5e-6, 0)   #[m]
        p2 = (2.9e-6, 2.5e-6, 0)  #[m]

        sys.add_line_defects([p1, p2], E, N, S)

        sys.finalize()
        return sys


A good way to start is by computing the thermal equilibrium electrostatic
potential. This will provide a guess for the drift diffusion Poisson solver
later on. Because of our geometry the potential on the left and right read

.. math::
   \phi(0, y) &= \frac{k_BT}{e}\ln\left(N_D/N_C \right)\\
   \phi(L, y) &= -E_g - \frac{k_BT}{e}\ln\left(N_A/N_V \right)

which is computed as follows::
    
    sys = system()
    v_left  = np.log(abs(sys.rho[0])/sys.Nc[0])
    v_right = -sys.Eg[0] - np.log(abs(sys.rho[sys.nx-1])/sys.Nv[sys.nx-1])

Observe the absolute value of the charge taken on the first line. This is
because the static charge there is negative (from the acceptors).
In order to solve the Poisson equation we need an initial guess (linear here)
and call the solver::

    # Initial guess
    v = np.empty((sys.nx,), dtype=float) 
    v[:sys.nx] = np.linspace(v_left, v_right, sys.nx)
    v = np.tile(v, sys.ny) # replicate the guess in the y-direction

    # Call Poisson solver with a tolerance of 10^-9
    v = sesame.poisson_solver(sys, v, 1e-9, info=1, max_step=100)

By default the solver assumes periodic boundary conditions in all directions
parallel to the contacts. One can change this setting to abrupt boundary
conditions by setting the flag ``periodic_bcs`` to ``False``.

We can now solve the drift diffusion Poisson equations to compute a
J(V) characteristics. The call to the drift diffusion Poisson solver returns a
dictionary with all values of electrostatic potential and quasi-Fermi levels. In
the following we solve the problem for multiple applied voltages and save the
output after each step::

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

The saving command is on the last line. This way of saving the data creates
multiple files like ``data.vapp_idx_1.npy`` containing a list of the 1D arrays of
the solution for the electron and hole quasi-Fermi levels, as well as the
electrostatic potential. 

While it is tempting to run the solver in parallel for each values of
applied voltage, the solver will fail with this approach. Note that the
results extracted after each step of the for loop are used as a new guess for
the next value of applied voltage. This method provides better chances to reach
convergence at each step. More about the solver can be found in the section
about the :ref:`algo`.

.. hint::
   In the case of an applied generation, it might be useful to perform
   calculations at zero bias under smaller generation amplitudes so that a good
   guess can be found. A similar approch can be used with the density of
   defects.

**Solvers options:** Both :func:`~sesame.solvers.poisson_solver` and
:func:`~sesame.solvers.ddp_solver` can make use of the MUMPS library if Sesame
was built against it. For that, pass the argument ``with_mumps=True`` to these
functions. For more information about the parameters used in the code above,
see the reference code :doc:`reference code <../reference/sesame.solvers>`.
