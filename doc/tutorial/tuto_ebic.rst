Tutorial 4: Simulating an EBIC/CL experiment
---------------------------------------------------------

In this tutorial we build a 2-dimensional simulation to describe experiments with a localized carrier generation profile, such as electron beam induced current (EBIC), or cathodoluminescence (CL).  In this case we'll need to define a system as before, and then define a custom carrier generation rate density profile.  We'll then cycle over beam positions and compute the total current and total radiative recombination as a function of beam position.

.. image:: ebic.*
   :align: center  

Building the system
........................

The system is built as before: a 2-dimensional p-n junction in which the "top" of the system represents the exposed sample surface.  Building the system works as in the previous tutorials, and the code shown below::


    	## dimensions of the system
	Lx = 3e-4   #[cm]
	Ly = 3e-4   #[cm]
	
	# extent of the junction from the left contact [cm]
	junction = .1e-4    # [cm]
	
	# Mesh
	x = np.concatenate((np.linspace(0,.2e-4, 30, endpoint=False),
	                    np.linspace(0.2e-4, 1.4e-4, 60, endpoint=False),
	                    np.linspace(1.4e-4, 2.7e-4, 60, endpoint=False),
	                    np.linspace(2.7e-4, Lx, 10)))
	
	y = np.concatenate((np.linspace(0, .25e-4, 50, endpoint=False),
	                    np.linspace(.25e-4, 1.25e-4, 50, endpoint=False),
	                    np.linspace(1.25e-4, Ly, 50)))
	
	# Create a system
	sys = sesame.Builder(x, y)
	
	# Dictionary with the material parameters
	mat = {'Nc':8e17, 'Nv':1.8e19, 'Eg':1.5, 'epsilon':9.4, 'Et': 0,
	       'mu_e':320, 'mu_h':40, 'tau_e':10*1e-9, 'tau_h':10*1e-9}
	
	# Add the material to the system
	sys.add_material(mat)
	
	# define a function specifiying the n-type region
	def region(pos):
	    x, y = pos
	    return x < junction
	# define a function specifiying the p-type region
	region2 = lambda pos: 1 - region(pos)
	
	# Add the donors
	nD = 1e17 # [cm^-3]
	sys.add_donor(nD, region)
	# Add the acceptors
	nA = 1e15 # [m^-3]
	sys.add_acceptor(nA, region2)
	
	# Use Ohmic contacts
	Sn_left, Sp_left, Sn_right, Sp_right = 1e7, 1e7, 1e7, 1e7
	sys.contact_S(Sn_left, Sp_left, Sn_right, Sp_right)
	sys.contact_type('Ohmic','Ohmic')


Adding surface recombination
............................

To add recombination at the sample surface, we add a planar defect along the line :math:`y=L_y`.  We consider a neutral surface, so that the charge of both transition states are 0.:: 

    p1 = (0, Ly)
    p2 = (Lx, Ly)

    E = 0                   # energy of gap state (eV) from intrinsic level
    rhoGB = 1e14            # density of defect states [cm^-2]
    s = 1e-14               # defect capture cross section [cm^2]

    sys.add_line_defects([p1, p2], rhoGB, s, E=E, transition=(0,0))

Electron beam excitation
............................

Next we consider the physics of the electron beam excitation.  Exciting the sample with an electron beam leads to a localized generation of charge carriers.  A very simple parameterization of the generation rate density profile is given as (add reference):

.. math:: 

   G(x,y) &= G_{\rm tot} \times \exp\left(-\frac{(x-x_0)^2+(y-y_0)^2}{2R_B^2}\right)\\
   G_{tot} &\approx \frac{I_{\rm beam}}{q} \times \frac{E_{\rm beam}}{3 E_g}\\
   R_B &= r_0 \left(\frac{0.043}{\rho/\rho_0}\right) \times \left(E_{\rm beam} /E_0\right)^{1.75}
   :label: Gxy

where :math:`r_0=1~{\rm \mu m},~\rho_0=1~{\rm g/cm^3},~E_0=1~{\rm keV}`.  The excitation is centered around the position :math:`(x_0,y_0)`.  :math:`x_0` is given by the lateral beam position, while the depth of the excitation from the sample surface is :math:`y_0=0.3 R_B`.  To code :math:`G(x,y)`, we start by making the necessary definitions::

	q = 1.6e-19      # C
	ibeam = 10e-12   # A
	Ebeam = 15e3     # eV
	eg = 1.5         # eV
	density = 5.85   # g/cm^3
	kev = 1e3        # eV
	
	Gtot = ibeam/q * Ebeam / (3*eg)			
	Rbulb = 0.043 / density * (Ebeam/kev)**1.75 	# given in micron
	Rbulb = Rbulb * 1e-4  				# converting to cm
	
	sigma = Rbulb / np.sqrt(15)		 	# Gaussian spread
	y0 = 0.3 * Rbulb				# penetration depth


Perfoming the beam scan
........................

To scan the lateral position :math:`x_0` of the beam, we first define the list of :math:`x_0` values::

	x0list = np.linspace(.1e-4, 2.5e-4, 11)

We define an array to store the computed current at each beam position::

	jset = np.zeros(len(x0list))
	
Next we scan over :math:`x_0'::

	for idx, x0 in enumerate(x0list):

Here we make the definition of :math:`G(x,y)` for a given value of :math:`x_0`, and add it to the system definition::	

	    def excitation(x,y):
	        return Gtot/(2*np.pi*sigma**2*Ld) * np.exp(-(x-x0)**2/(2*sigma**2)) 	* np.exp(-(y-Ly+y0)**2/(2*sigma**2))
	
	    sys.generation(excitation)
	
Now we solve the system::

	    solution = sesame.solve(sys, periodic_bcs=False, tol=1e-8)

We obtain the current and store it in the array::
	
	    # get analyzer object with which to compute the current
	    az = sesame.Analyzer(sys, solution)
	    # compute (dimensionless) current and convert to dimension-ful form
	    tj = az.full_current() * sys.scaling.current * sys.scaling.length
	    # save the current
	    jset[idx] = tj

