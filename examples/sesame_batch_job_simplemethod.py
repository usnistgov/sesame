#!/usr/bin/env python3

import numpy as np
import sesame
import itertools
from mpi4py import MPI

def system(rho_GB, E_GB, S_GB, tau):

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
    rho_GBlist = [1e11, 1e12]           # [1/cm^2]
    E_GBlist = [-.3, .3]                  # [eV]
    S_GBlist = [1e-15, 1e-16]           # [cm^2]
    taulist = [1e-7, 1e-8]             # [s]

    # Specify applied voltages
    voltages = np.linspace(0, .1, 2)

    # Define array to store computed J-V values
    jvset_local = np.zeros([len(rho_GBlist),len(E_GBlist),len(S_GBlist),len(taulist),len(voltages)])
    jvset = np.zeros([len(rho_GBlist),len(E_GBlist),len(S_GBlist),len(taulist),len(voltages)])

    jobcounter = 0

    # Vary all parameters
    for E_GBcounter in range(0,len(E_GBlist)):
        E_GB = E_GBlist[E_GBcounter]

        for rho_GBcounter in range(0, len(rho_GBlist)):
            rho_GB = rho_GBlist[rho_GBcounter]

            for S_GBcounter in range(0, len(S_GBlist)):
                S_GB = S_GBlist[S_GBcounter]

                for taucounter in range(0, len(taulist)):
                    tau = taulist[taucounter]

                    jobcounter = jobcounter + 1

                    # Divide jobs among the processors
                    if (np.mod(jobcounter, mpisize) == mpirank):

                        # Get system for given set of parameters
                        sys = system(rho_GB, E_GB, S_GB, tau)

                        # Get equilibrium solution
                        eqsolution = sesame.solve_equilibrium(sys)

                        # Define a function for generation profile
                        f = lambda x, y: 2.3e21 * np.exp(-2.3e4 * x)
                        # add generation to the system
                        sys.generation(f)

                        # Specify output filename for given parameter set
                        outputfile = 'EGB{0:d}_rhoGB{1:d}_SGB{2:d}_tau{3:d}'.format(E_GBcounter,rho_GBcounter,S_GBcounter,taucounter)
                        # Compute J-V curve
                        jv = sesame.IVcurve(sys, voltages, eqsolution, outputfile)
                        # Save computed J-V in array
                        jvset_local[E_GBcounter, rho_GBcounter, S_GBcounter, taucounter,:] = jv


    # Gather results from all processors
    mpi_comm.Reduce(jvset_local,jvset)

    # Save J-V data for all parameter sets
    np.savez("JVset",jvset)


