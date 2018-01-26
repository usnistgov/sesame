Installation instructions for beginners
---------------------------------------

This section documents how to build Sesame for those with zero python experience.  

Installing Python
+++++++++++++++++++++++++++

For users with no python installation, there are a number of convenient standalone installations which automatically includes all of the requisiste libraries and packages, including:

* `Anaconda <https://www.anaconda.com/>`_ 
* `Canopy <https://www.enthought.com/product/canopy/>`_
* `Pythonxy <https://python-xy.github.io/>`_

These can be installed on any operating system (windows, linux, macOS).  This page walks through the process using Anaconda in a Windows environment.


First download and install Anaconda, using the default settings.  After installation, you'll find a new folder with various programs in the windows ``Start`` button folder: ``Start`` :math:`\rightarrow` ``All Programs`` :math:`\rightarrow` ``Anaconda``:.  

Downloading and Installing Sesame (on Windows)
++++++++++++++++++++++++++++++++++++++++++++++

To obtain Sesame, first open the Anaconda Prompt: ``Start`` :math:`\rightarrow` ``All Programs`` :math:`\rightarrow` ``Anaconda`` :math:`\rightarrow` ``Anaconda Prompt``.  A command line should appear (a primer on using the Windows command line can be found `here <https://www.computerhope.com/issues/chusedos.htm>`_).  Sesame is downloaded with the command::

	git sesame

(Have to figure out where git where put the source code).  Build and install Sesame with the commands::

    python setup.py build
    python setup.py install --user

The essential procedure for installing for other operating systems is the same.  

Running Sesame
+++++++++++++++++++++++++++++++++
Upon installation, you can try some of the examples.  Navigate to the `examples` directory::

	cd sesame\examples

Running a sesame python script is done via:

	python 1dpn.py

