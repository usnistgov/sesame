Tutorial 2: IV curve of a one-dimensional *pn* heterojunction
---------------------------------------------------------

In this tutorial we consider a more complex system in 1-dimension: a heterojunction with a Schottky back contact.  The n-type material is CdS and the p-type material is CdTe.  The structure of the script is the same as in the last tutorial, however we must provide more detail to describe a more complex system.  The band diagram for this 
system under short-circuit conditions is shown below
.

.. image:: 1d_hetero.*
   :align: center

Constructing a grid and building the system
...........................................

We first define the thicknesses of the n-type and p-type regions::

    t1 = 25*1e-7    # thickness of CdS
    t2 = 4*1e-4     # thickness of CdTe


The mesh for a heterojunction should be very fine in the immediate vicinity of the materials interface.  We define a distance ``dd``, which determines the thickness of the highly-refined mesh near an interface.  We form the overall system mesh by concatenating meshes for different parts of the system as follows::

    	dd = 3e-6 # 2*dd is the distance over which mesh is refined
	x = np.concatenate((np.linspace(0, dd, 10, endpoint=False),                # L contact interface
                    np.linspace(dd, t1-dd, 50, endpoint=False),                    # material 1
                    np.linspace(t1 - dd, t1 + dd, 10, endpoint=False),             # interface 1
                    np.linspace(t1 + dd, (t1+t2) - dd, 100, endpoint=False),       # material 2
                    np.linspace((t1+t2) - dd, (t1+t2), 10)))                       # R contact interface



As before we make a system with :func:`~sesame.builder.Builder`::

    sys = sesame.Builder(x)


Adding material properties
...........................................

We make functions to define the two regions.  For this tutorial, we'll use the ``lambda`` keyword to efficiently write the functions (briefly, the input argument is defined between "lambda" and ":", and the function instruction is given after the ":" symbol)::

    region1 = lambda x: x<=t1   # CdS region
    region2 = lambda x: x>t1    # CdTe region


Now we need to add materials to our system.  We have two dictionaries to describe the two material types.  As in the last tutorial, we add materials using the ``sys`` method :func:`~sesame.builder.Builder.add_material`.  However we provide additional input arguments to :func:`~sesame.builder.Builder.add_material` to specify the material location::

    CdS = {'Nc': 2.2e18, 'Nv':1.8e19, 'Eg':2.4, 'epsilon':10, 'Et': 0,
        'mu_e':100, 'mu_h':25, 'tau_e':1e-8, 'tau_h':1e-13, 'affinity': 4.}
    CdTe = {'Nc': 8e17, 'Nv': 1.8e19, 'Eg':1.5, 'epsilon':9.4, 'Et': 0,
        'mu_e':320, 'mu_h':40, 'tau_e':5e-9, 'tau_h':5e-9, 'affinity': 3.9}

    sys.add_material(CdS, region1)     # adding CdS
    sys.add_material(CdTe, region2)     # adding CdTe

In this example we've explicitly specifed the recombination defect energy level ``Et``.  As described in the last tutorial, this energy level is specified with respect to the intrinsic energy level.  

Adding dopants
...........................................

Adding the dopants works as in the last tutorial::

    
    nD = 1e17  # donor density [cm^-3]
    sys.add_donor(nD, region1)
    nA = 1e15  # acceptor density [cm^-3]
    sys.add_acceptor(nA, region2)


Specifying contact types
...........................................

Next, we'll add a left Ohmic contact and a right Schottky contact.  For Schottky contacts, it's necessary to specify the work function of the metal.  For Ohmic contacts, the metal work function doesn't enter into the problem, so its value is unimportant - we'll therefore simply set the left contact work function equal to 0.  As in the previous tutorial, we add contacts to the system using the ``sys`` method :func:`~sesame.builder.Builder.contact_type`; however this time we provide the additional arguments of the left and right contact work functions to :func:`~sesame.builder.Builder.contact_type`::

    Lcontact_type, Rcontact_type = 'Ohmic', 'Schottky'
    Lcontact_workFcn, Rcontact_workFcn = 0, 5.0   

    sys.contact_type(Lcontact_type, Rcontact_type, Lcontact_workFcn, Rcontact_workFcn)

Having defined the contact types, we next specify the contact recombination velocities as before::

    Sn_left, Sp_left, Sn_right, Sp_right = 1e7, 1e7, 1e7, 1e7
    sys.contact_S(Sn_left, Sp_left, Sn_right, Sp_right)


Computing an I-V curve
......................
We've now completed the system definition.  As in the last example, we compute the equilibrium solution, add illumination, and compute the IV curve

.. warning::
   Sesame does not include interface current mechanisms of       thermionic emission and tunneling.

:: 

    eqsolution = sesame.solve_equilibrium(sys)

    phi = 1e21 # photon flux [1/(m^2 s)]
    alpha = 2.3e6 # absorption coefficient [1/m]

    # Define a function for the generation rate
    f = lambda x: phi * alpha * np.exp(-alpha * x)
    sys.generation(f)

    voltages = np.linspace(0, 0.95, 40)
    j = sesame.IVcurve(sys, voltages, eqsolution, '1dhetero_IV')

The current can be saved and plotted as in the previous tutorial::

    np.save('jv1d_hetero',result)  
    import matplotlib.pyplot as plt
    plt.plot(voltages,j,'-o')
    plt.xlabel('Voltage [V]')
    plt.ylabel('Current [A/cm^2]')
    plt.grid()
    plt.show()


.. image:: jv_hetero.*
   :align: center