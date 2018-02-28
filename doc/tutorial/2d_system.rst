Tutorial 3: Two-dimensional *pn* junction with a grain boundary
---------------------------------------------------------------
In this tutorial we show how to build a two-dimensional pn junction containing a
grain boundary. We focus on the creation of different regions and the addition of
extra charges to the system. See :doc:`tutorial 1 <tuto1>` for the basics on the
creation of systems. We present the tools available to vizualize the system
created.

.. seealso:: The example treated here is in the file ``2dpn.py`` in the
   ``examples`` directory in the root directory of the distribution. 

Building a two-dimensional system
...............................................
Suppose we want to simulate a two-dimensional pn junction (homojunction) with a
line of defects as depicted below.  

.. image:: geometry.*
   :align: center

As usual, we start by importing the sesame package and numpy. We construct the
mesh of the system and make an instance of the :func:`~sesame.builder.Builder`.  Notice that in this case we provide the ``Builder`` function with both x and y grids; this automatically tells the code to build a two-dimensional system::

    import sesame
    import numpy as np

    # dimensions of the system
    Lx = 3e-4 # [cm]
    Ly = 3e-4 # [cm]

    # extent of the junction from the left contact [m]
    junction = 10e-9 

    # Mesh
     x = np.concatenate((np.linspace(0, .2e-4, 30, endpoint=False),
                    np.linspace(0.2e-4, 1.4e-4, 60, endpoint=False),
                    np.linspace(1.4e-4, 2.7e-4, 60, endpoint=False),
                    np.linspace(2.7e-4, 2.98e-4, 30, endpoint=False),
                    np.linspace(2.98e-4, Lx, 10)))

     y = np.concatenate((np.linspace(0, 1.25e-4, 60, endpoint=False),
                    np.linspace(1.25e-4, 1.75e-4, 50, endpoint=False),
                    np.linspace(1.75e-4, Ly, 60)))

    # Create a system
    sys = sesame.Builder(x, y)

We define and add a material as before::

    # Dictionary with the material parameters
    reg1 = {'Nc':8e17, 'Nv':1.8e19, 'Eg':1.5, 'epsilon':9.4,
            'mu_e':200, 'mu_h':200, 'tau_e':10e-9, 'tau_h':10e-9}

    # Add the material to the system
    sys.add_material(reg1)

We next define functions delimiting the regions with different doping values. Because the model is 2-dimensional, there's a slight difference with previous tutorials.  The function input arguments are now ``tuples``, of the form ``(x,y)``.  In the function, we first "unpack" the x and y coordinate, and determine the location of the x-coordinate relative to the junction::

    # Add the donors
    def n_region(pos):
        x, y = pos
        return x < junction
    nD = 1e17  # [cm^-3]
    sys.add_donor(nD, n_region)

    # Add the acceptors
    def p_region(pos):
        x, y = pos
        return x >= junction    
    nA = 1e15  # [cm^-3]
    sys.add_acceptor(nA, p_region)

We specify contacts as before::


    # Use perfectly selective Ohmic contacts
    sys.contact_type('Ohmic', 'Ohmic')
    Sn_left, Sp_left, Sn_right, Sp_right = 1e7, 0, 0, 1e7
    sys.contacts(Sn_left, Sp_left, Sn_right, Sp_right)

.. note::
    Sesame assumes that the contact properties (e.g. recombination velocity, metallic work function) are uniform along the y-direction.

Adding a grain boundary
........................

Now we add a line of defects to simulate a grain boundary. We
define a defect gap state as follows::

    # gap state characteristics
    s = 1e-15                # trap capture cross section [m^2]
    E = -0.25                # energy of gap state (eV) from intrinsic energy level
    N = 2e13                 # defect density [1/m^2]

    # Specify the two points that make the line containing additional charges
    p1 = (20e-7, 2.5e-4)   # [cm]
    p2 = (2.9e-4, 2.5e-4)  # [cm]

    # Pass the information to the system
    sys.add_line_defects([p1, p2], N, s, E=E, transition=(1/-1))

The type of the charge transition :math:`\alpha/\beta` is specified as
shown above. In our example we chose a mixture of donor and acceptor at energy
E. An acceptor would be described by (-1,0) and a donor by (1,0).

.. note::
   * Avoid adding charges on the contacts of the system, as these will not be
     taken into account. The code is not equiped to deal with such boundary
     conditions.
   * In order to add another gap state at a different energy at the same
     location, one repeats the exact same process.  
   * Here we assumed equal electron and hole surface recombination velocities.
     The function :func:`~sesame.builder.Builder.add_line_defects` takes two
     surface recombination velocities as argument. The first is for electrons,
     the second for holes. To use different values write

     .. code-block:: python

        sys.add_line_defects([p1, p2], N, sn, sp, E=E)
   * A continuum of states can be considered by omitting the energy argument
     above. The density of states can be a callable function or a numerical
     value, in which case the density of states is independent of the energy.


Computing the IV curve
........................

The computation of the IV curve proceeds as in the previous tutorials.  We show the code below::

  # Specify applied voltages
  voltages = np.linspace(0,1,20)
  # Compute IV curve
  j = sesame.IVcurve(sys, voltages, solution, 'GB_JV')
  # Save the computed IV data
  result = {'voltages':voltages,'j':j}
  np.save('2dGB_IV',result)






Plotting system variables
..........................

The solution can be visualized using matplotlib, as discussed more fully in tutorial 5.  Here we give a brief example of loading an output file and plotting the electrostatic potential versus position.  First we import the plotting library, and load one of the files saved in the IVcurve function::

  import matplotlib.pyplot as plt
  # Load an outputfile file and do a surface plot of v
  results = np.load('GB_JV_0.npz')

The ``results`` variable contains information about the system and the solution.  Refer to :doc:`tutorial 5 <analysis>` for details on how to plot and analyze the solution.  the following code generates a 2-d contour plot of the electrostatic potential::

  # need to reshape the potential from a 1-d array to a 2-d array
  v = np.reshape(results['v'],[sys.ny,sys.nx])
  # rescale the potential to dimension-ful form
  v = v * sys.scaling.energy
  # make contour plot 
  plt.contourf(sys.xpts,sys.ypts,v)
  plt.xlabel('Position [cm]')
  plt.ylabel('Position [cm]')
  plt.colorbar()
  plt.title('V')
  plt.show()

The output is shown below:

.. image:: GB_potential.*
   :align: center

Spatial variation of material parameters
..........................................


Suppose we want to have a reduced mobility around the line defects compared to the rest
of the system. Therefore we need to define two regions in our system, two large
regions with mobility :math:`200~ \mathrm{cm^2/(V\cdot s)}` and a smaller one
around the line defect with mobility :math:`20~\mathrm{cm^2/(V\cdot s)}`. 



In the definition of ``region1``, observe how we define the statement OR. Here
we use a bitwise logical operator. Other useful operators are ``&`` for AND,
``~`` for NOT. Statements on each side of an operator must be in between
parentheses.  We can easily use ``region1`` to define the second region, since
all sites not in region 1 will be in region 2::

   


