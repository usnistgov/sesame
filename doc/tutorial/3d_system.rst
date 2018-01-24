Tutorial 5: Plane defects in three-dimensional systems
-------------------------------------------------------


In this short tutorial we show how to add plane defects in a 3D system. Sesame
handles planes that are rectangular and with a least two edges parallel to a
main axis.

.. seealso:: The example treated here is in the file ``3dpn.py`` in the
   ``examples`` directory in the root directory of the distribution. 

As shown below, planes are defined by four points. The first two points define
an edge of the plane, and the second two points define the perpendicular edge of
the rectangle::

    # dimensions of the system
    Lx = 3e-6      # [m]
    Ly = 5e-6      # [m]
    Lz = 2e-6      # [m]

    ...

    # gap state characteristics
    s = 1e-15 * 1e-4         # trap capture cross section [m^2]
    E = -0.25                # energy of gap state (ev) from midgap
    N = 2e13 * 1e4           # defect density [1/m^2]

    p1 = (1e-6, .5e-6, 1e-6)            # [m]
    p2 = (2.9e-6, .5e-6, 1e-6)          # [m]

    q1 = (1.0e-6, 4.5e-6, 1e-9)         # [m]
    q2 = (2.9e-6, 4.5e-6, 1e-9)         # [m]

    # pass the information to the system
    sys.add_plane_defects([p1, p2, q1, q2], N, S, E=E)

We can visualize the plane to make sure it resembles what we expect::

    sesame.plot_plane_defects(sys)


.. image:: 3dpn.*
   :align: center
