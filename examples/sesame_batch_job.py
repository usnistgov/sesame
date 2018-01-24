#!/usr/bin/env python3

import numpy as np
import sesame
import itertools
from mpi4py import MPI

def system(params):

    # we assume the params are given in order: [rho_GB, E_GB, S_GB, tau]
    rho_GB = params[0]
    E_GB = params[1]
    S_GB = params[2]
    tau = params[3]

    # Dimensions of the system
    Lx = 3e-4  # [cm]
    Ly = 3e-4  # [cm]

    # Mesh
    x = np.concatenate((np.linspace(0, 0.2e-4, 30, endpoint=False),
                        np.linspace(0.2e-4, 1.4e-4, 60, endpoint=False),
                        np.linspace(1.4e-4, 2.7e-4, 60, endpoint=False),
                        np.linspace(2.7e-4, 2.98e-4, 30, endpoint=False),
                        np.linspace(2.98e-4, Lx, 10)))

    y = np.concatenate((np.linspace(0, 1.25e-4, 60, endpoint=False),
                        np.linspace(1.25e-4, 1.75e-4, 50, endpoint=False),
                        np.linspace(1.75e-4, Ly, 60)))

    # Create a system
    sys = sesame.Builder(x, y)

    # Dictionary with the material parameters
    mat = {'Nc': 8e17, 'Nv': 1.8e19, 'Eg': 1.5, 'epsilon': 9.4, 'Et': 0,
           'mu_e': 320, 'mu_h': 40, 'tau_e': tau, 'tau_h': tau}

    # Add the material to the system
    sys.add_material(mat)

    # Extent of the junction from the left contact [cm]
    junction = .1e-4  # [cm]

    # Define a function specifiying the n-type region
    def region1(pos):
        x, y = pos
        return x < junction

    # Define a function specifiying the p-type region
    def region2(pos):
        x, y = pos
        return x >= junction

    # Add the donors
    nD = 1e17  # Donor density [cm^-3]
    sys.add_donor(nD, region1)
    # Add the acceptors
    nA = 1e15  # Acceptor density [cm^-3]
    sys.add_acceptor(nA, region2)

    # Define contacts: CdS and CdTe contacts are Ohmic
    sys.contact_type('Ohmic', 'Ohmic')
    # Define the surface recombination velocities for electrons and holes [cm/s]
    Sn_left, Sp_left, Sn_right, Sp_right = 1e7, 1e7, 1e7, 1e7
    # This function specifies the simulation contact recombination velocity [cm/s]
    sys.contact_S(Sn_left, Sp_left, Sn_right, Sp_right)

    # Specify the two points that make the line containing additional charges
    p1 = (0.1e-4, 1.5e-4)    # [cm]
    p2 = (2.9e-4, 1.5e-4)    # [cm]

    # Add donor defect along GB
    sys.add_line_defects([p1, p2], rho_GB, S_GB, E=E_GB, transition=(1, 0))
    # Add acceptor defect along GB
    sys.add_line_defects([p1, p2], rho_GB, S_GB, E=E_GB, transition=(0, -1))

    return sys


if __name__ == '__main__':

    # Initiate MPI
    mpi_comm = MPI.COMM_WORLD
    mpirank = mpi_comm.Get_rank()
    mpisize = mpi_comm.Get_size()

    # Set of parameters to vary - these parameters defines 180 simulations
    rho_GBlist = [1e11, 1e12, 1e13]           # [1/cm^2]
    E_GBlist = [-.3, 0, .3]                  # [eV]
    S_GBlist = [1e-14, 1e-15, 1e-16]           # [cm^2]
    taulist = [1e-7, 1e-8, 1e-9]             # [s]

    # Specify applied voltages
    voltages = np.linspace(0, .1, 2)

    # this function generates all sets of parameter sets from the constituent lists
    paramlist = list(itertools.product(rho_GBlist, E_GBlist, S_GBlist, taulist))
    njobs = len(paramlist)

    # Define array to store computed J-V values
    jvset_local = np.zeros([njobs, len(voltages)])
    jvset = np.zeros([njobs, len(voltages)])

    # or params = [rho_GBlist, E_GBlist, S_GBlist, taulist]
    # paramlist = itertools.product(*params)

    my_param_indices = np.arange(mpirank,njobs,mpisize)

    # cycle over all parameter sets
    for myjobcounter in my_param_indices:

        # Get system for given set of parameters
        params = paramlist[myjobcounter]
        sys = system(params)

        # Get equilibrium solution
        #eqsolution = sesame.solve_equilibrium(sys)

        # Define a function for generation profile
        f = lambda x, y: 2.3e21 * np.exp(-2.3e4 * x)
        # add generation to the system
        sys.generation(f)

        # Specify output filename for given parameter set
        outputfile = ''
        for paramvalue in params:
            outputfile = outputfile + '{0}_'.format(paramvalue)

        # Compute J-V curve
        jv = sesame.IVcurve(sys, voltages, eqsolution, outputfile)
        # Save computed J-V in array
        jvset_local[myjobcounter,:] = jv


    # Gather results from all processors
    mpi_comm.Reduce(jvset_local,jvset)

    # Save J-V data for all parameter sets
    np.savez("JVset", jvset, paramlist)


