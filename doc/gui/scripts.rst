Running scripts in the GUI
---------------------------

The standalone Sesame GUI executable contains a full Python distribution.  This can be accessed by selecting the ``Console`` :math:`\rightarrow` ``Show Console`` option from the window.  An IPython console is displayed, as shown below.  Below we show how Sesame scripts may be run from the python console:

.. image:: ipython.*
   :align: center

The first step is to navigate to the directory with the python script you would like to run.  This is accomplished with the following commands, entered into the IPython prompt::

    import os
    os.chdir("examples/tutorial1")

Next the python script (in this case called "1d_homojunction.py") is called with the following command::

    exec(open("1d_homojunction.py").read())

The example scripts and GUI input files provided in the examples/tutorials folders define identical simulations.  The user can modify one or the other as needed, and run either one within the GUI.  Scripts provide more flexibility for system definition and for simulation actions, e.g. looping over several variables.

