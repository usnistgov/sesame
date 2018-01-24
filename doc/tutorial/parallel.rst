Tutorial 7: Batch submission for computing clusters
---------------------------------------

Running sesame on a cluster
............................

Next we give an example of running Sesame on a computing cluster.  This can be accomplished in several different ways, and the most efficient way depends on the details of the cluster environment.  For this example, we use the MPI library.  This determines how sesame is called from the command line, and how the parallel-ization is implemented in the script.  The script "parallel_batch_example.py" is run on 32 processors with the following command::

	mpirun -np 32 python3 parallel_batch_example.py

Parallel script description
.............................
We first import some additional packages::

	import numpy as np
	import sesame
	import itertools
	from mpi4py import MPI

Defining the system
....................

We define a function which takes in a list of parameters.  In this case, the parameters are :math:`\rho_{GB},~E_{GB},~S_{GB},~\tau`.  The construction of the system follows the previous examples::

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
	
	    sys = sesame.Builder(x, y)	    # Create a system
	
	    # Dictionary with the material parameters
	    mat = {'Nc': 8e17, 'Nv': 1.8e19, 'Eg': 1.5, 'epsilon': 9.4, 'Et': 0,
	           'mu_e': 320, 'mu_h': 40, 'tau_e': tau, 'tau_h': tau}
	
	    sys.add_material(mat)	    # Add the material to the system

	    junction = .1e-4  # [cm]
	
	    # Define a function specifiying the n-type region
	    def region1(pos):
	        x, y = pos
	        return x < junction
	
	    # Define a function specifiying the p-type region
	    def region2(pos):
	        x, y = pos
	        return x >= junction
	
	    nD = 1e17  				# Donor density [cm^-3]
	    sys.add_donor(nD, region1)	    	# Add the donors
	    nA = 1e15  				# Acceptor density [cm^-3]
	    sys.add_acceptor(nA, region2)	    	# Add the acceptors

	
	    # Define contacts: CdS and CdTe contacts are Ohmic
	    sys.contact_type('Ohmic', 'Ohmic')
	    Sn_left, Sp_left, Sn_right, Sp_right = 1e7, 1e7, 1e7, 1e7
	    sys.contact_S(Sn_left, Sp_left, Sn_right, Sp_right)
	
	    # Specify the two points that make the line containing additional 	charges
	    p1 = (0.1e-4, 1.5e-4)    # [cm]
	    p2 = (2.9e-4, 1.5e-4)    # [cm]
	
	    # Add donor defect along GB
	    sys.add_line_defects([p1, p2], rho_GB, S_GB, E=E_GB, transition=(1, 0))
	    # Add acceptor defect along GB
	    sys.add_line_defects([p1, p2], rho_GB, S_GB, E=E_GB, transition=(0, -1))

	    return sys



Cycling over parameters on a computing cluster
.............................................................

Next we give an example of how to distribute the jobs among an arbtirary number of processors.  Instead of a python script, this program is the python version of an executable.  This is ::

	if __name__ == '__main__':

Inside the main program, we first initialize the MPI library with the command::
	
	    mpi_comm = MPI.COMM_WORLD


Next each processor finds their "rank" (explain)::


	    mpirank = mpi_comm.Get_rank()
	    mpisize = mpi_comm.Get_size()

We define the set of parameter lists we want to study::
	
	    # Set of parameters to vary - these parameters defines 180 simulations
	    rho_GBlist = [1e11, 1e12, 1e13]           # [1/cm^2]
	    E_GBlist = [-.3, 0, .3]                  # [eV]
	    S_GBlist = [1e-14, 1e-15, 1e-16]           # [cm^2]
	    taulist = [1e-7, 1e-8, 1e-9]             # [s]

we use this convenient thing::

	
	    # this function generates all sets of parameter sets from the 	constituent lists
	    paramlist = list(itertools.product(rho_GBlist, E_GBlist, S_GBlist, 	taulist))

we find the total number of simulations.  This is equal to the product of the length of all parameter lists.  This can get quite large if we vary several parameters::

	    njobs = len(paramlist)

Here's where the parallel-ization of the batch processes enters.  Each node only needs to compute a subset of all parameters.  We set the relevant parameters for each node as follows::

	    my_param_indices = np.arange(mpirank,njobs,mpisize)

We define two arrays in which to store the computed J-V values.  One of them is a local array, the other is a "global" array into which all the computed values will be set at the end of the program::
	
	    # Define array to store computed J-V values
	    jvset_local = np.zeros([njobs, len(voltages)])
	    jvset = np.zeros([njobs, len(voltages)])
	
Here we define the set of applied voltages::	

	    # Specify applied voltages
	    voltages = np.linspace(0, .1, 2)

Now we cycle over all the parameter sets which apply to a given node::
	
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



