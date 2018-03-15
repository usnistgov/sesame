Analysis Tab
----------------

After successful completion of the simulation, the ``Analysis tab`` provides tools to plot and analyze the solution.  



.. image:: analysis.*
   :align: center

**Import data**
The upper-left frame lets the user specify the output files to analyze.  This list of files will be automatically populated by the output files of a successful simulation specified in the ``Simulation tab``.   

**Surface plot**
For plots of quantities in 2-dimensions, select an outputfile, and in the ``surface plot``, select a quantity to plot.  Clicking ``Plot`` will give the spatially resolved color contour plot of the given quantity.  

**Linear plot**
For 1-d plots, two options are available: "loop values" and "position".  For "loop values", a single scalar quantity (e.g. current density, total recombination) is plotted against the values of the looped parameter.  Selecting "loop values" automatically selects all output files, and automatically fills in the X data field with the loop values.  This is shown in the graphic above.

If "Position" is selected, then a plot of a quantitiy versus position is given.  For a 2-d simulation, the position is given by two endpoints of a line (note the position of the endpoints are assumed to be given in units of [cm]).  This is specified by the user in the "S data" field.  For a 1-d simulation, the "X data" field is automatically set to the entire 1-d grid.  

