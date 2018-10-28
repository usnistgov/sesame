# Copyright 2017 University of Maryland.
#
# This file is part of Sesame. It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution.

from scipy.interpolate import InterpolatedUnivariateSpline as spline
import numpy as np
import scipy.constants as cts
from collections import namedtuple
from itertools import product
import warnings

from . import utils

import logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

# named tuple of the characteristics of a defect
defect = namedtuple('defect', ['sites', 'location', \
                               'dos', 'energy', 'sigma_e', 'sigma_h',\
                               'transition', 'perp_dl'])

class Scaling():
    """
    An object defining the scalings of the drift-diffusion-Poisson equation. The
    temperature of the system and the reference unit for lengths are specified
    when an instance is created. The default unit for length is cm, and the
    default temperature is 300 K.

    Parameters
    ----------
    input_length: string
        Reference unit for lengths. Acceptable entries are 'cm' for centimeters
        and 'm' for meters.
    T: float
        Temperature for the simulation.

    Attributes
    ----------
    denisty: float
        Density scale taken equal to 10\ :sup:`19` cm\ :sup:`-3`.
    energy: float
        Energy scale.
    mobility: float
        Mobility scale taken equal to 1 cm\ :sup:`2`/(V.s).
    time: float
        Time scale.
    length: float
        Length scale.
    generation: float
        Scale of generation and recombination rates.
    velocity: float
        Velocity scale.
    current: float
        Electrical current density scale.
    """
    def __init__(self, input_length='cm', T=300):
        # densities
        if input_length == "m":
            self.density = 1e19 * 1e6 # [m^-3]
        else:
            self.density = 1e19 # [cm^-3]
        # energies
        self.energy = cts.k * T / cts.e
        # mobilities
        if input_length == "m":
            self.mobility = 1 * 1e-4 # [m^2 / (V.s)]
        else:
            self.mobility = 1 # [cm^2 / (V.s)]
        # time [s]
        if input_length == "m":
            self.time = cts.epsilon_0 / (self.mobility * cts.e * self.density) # s
        else:
            self.time = cts.epsilon_0*1e-2 / (self.mobility * cts.e * self.density)  # s
        # lengths 
        if input_length == "m":
            self.length = np.sqrt(cts.epsilon_0 * self.energy / (cts.e * self.density)) # m
        else:
            self.length = np.sqrt(cts.epsilon_0*1e-2 * self.energy / (cts.e * self.density))  # cm
        # generation rate [m^-3 s^-1]
        self.generation = (self.density * self.mobility * self.energy) / self.length**2 
        # recombination velocities
        self.velocity = self.length / self.time
        # current
        self.current = cts.k * T * self.mobility * self.density / self.length



class Builder():
    """
    A system discretized on a mesh. 

    This type discretizes a system on a mesh provided by the user, and takes
    care of all normalizations. The temperature of the system is specified when
    an instance is created. The default is 300 K. 

    Parameters
    ----------
    xpts, ypts, zpts: numpy arrays of floats
        Mesh with original dimensions.
    input_length: string
        Reference unit for lengths. Acceptable entries are 'cm' for centimeters
        and 'm' for meters.
    T: float
        Temperature for the simulation.


    Attributes
    ----------
    xpts, ypts, zpts: numpy arrays of floats
        Mesh with original dimensions.
    dx, dy, dz: numpy arrays of floats
        Dimensionless lattice constants in the x, y, z directions.
    nx, ny: integers
        Number of lattice nodes in the x, y directions.
    Nc, Nv: numpy arrays of floats
        Dimensionless effective densities of states of the conduction and
        valence bands.
    Eg: numpy array of floats
        Dimensionless band gap.
    ni: numpy array of floats
        Dimensionless intrinsic density.
    mu_e, mu_h:  numpy arrays of floats
        Dimensionless mobilities of electron and holes.
    tau_e, tau_h:  numpy arrays of floats
        Dimensionless bulk lifetime for electrons and holes.
    n1, p1:  numpy arrays of floats
        Dimensionless equilibrium densities of electrons and holes at the bulk trap state.
    bl: numpy array of floats
        Electron affinity.
    g: numpy array of floats
        Dimensionless generation for each site of the
        system. This is defined only if a generation profile was provided when
        building the system.
    gtot: float
        Dimensionless integral of the generation rate.
    defects_list: list of named tuples
        List of named tuples containing the characteristics of the defects in the
        order they were added to the system. The field names are sites,
        location, dos, energy, sigma_e, sigma_h, transition, perp_dl.
    """


    def __init__(self, xpts, ypts=np.zeros(1), input_length='cm', T=300, periodic=True):

        self.scaling = Scaling(input_length, T)
        self.input_length = input_length


        self.xpts = xpts
        self.dx = (self.xpts[1:] - self.xpts[:-1]) / self.scaling.length
        self.nx = xpts.shape[0]

        self.ypts = ypts
        self.dy = (self.ypts[1:] - self.ypts[:-1]) / self.scaling.length

        if len(self.dy) == 0:
            self.dy = np.append(self.dy, self.dx[0])
            self.dimension = 1
        else:
            self.dimension = 2
            if periodic is True:
                self.dy = np.append(self.dy, self.dy[0])
            else:
                self.dy = np.append(self.dy, np.inf)

        self.ny = ypts.shape[0]

        nx, ny = self.nx, self.ny
        self.Nc      = np.zeros((nx*ny,), dtype=float)
        self.Nv      = np.zeros((nx*ny,), dtype=float)
        self.Eg      = np.zeros((nx*ny,), dtype=float)
        self.epsilon = np.zeros((nx*ny,), dtype=float)
        self.mass_e  = np.zeros((nx*ny,), dtype=float)
        self.mass_h  = np.zeros((nx*ny,), dtype=float)
        self.mu_e    = np.zeros((nx*ny,), dtype=float)
        self.mu_h    = np.zeros((nx*ny,), dtype=float)
        self.tau_e   = np.zeros((nx*ny,), dtype=float)
        self.tau_h   = np.zeros((nx*ny,), dtype=float)
        self.n1      = np.zeros((nx*ny,), dtype=float)
        self.p1      = np.zeros((nx*ny,), dtype=float)
        self.bl      = np.zeros((nx*ny,), dtype=float)
        self.rho     = np.zeros((nx*ny,), dtype=float)
        self.g       = np.zeros((nx*ny,), dtype=float)
        self.B       = np.zeros((nx*ny,), dtype=float)
        self.Cn      = np.zeros((nx*ny,), dtype=float)
        self.Cp      = np.zeros((nx*ny,), dtype=float)
        self.Etrap   = np.zeros((nx*ny,), dtype=float)

        self.defects_list = []

    def add_material(self, mat, location=lambda pos: True):
        """
        Add a material to the system.

        Parameters
        ----------
        mat: dictionary 
            Contains the material parameters Keys are Nc (Nv): conduction
            (valence) effective densities of states [cm\ :sup:`-3`], Eg: band
            gap [:math:`\mathrm{eV}`], epsilon: material's permitivitty, mu_e
            (mu_h): electron (hole) mobility [m\ :sup:`2`/(V s)], tau_e (tau_h):
            electron (hole) bulk lifetime [s], Et: energy level of the bulk
            recombination centers [eV], affinity: electron affinity [eV], B:
            radiation recombination constant [cm\ :sup:`3`/s], Cn (Cp): Auger
            recombination constant for electrons (holes) [cm\ :sup:`6`/s],
            mass_e (mass_h): effective mass of electrons (holes). All parameters
            can be scalars or callable functions similar to the location
            argument.
        location: Boolean function
            Definition of the region containing the material. This function must
            take a tuple of real world coordinates (e.g. (x, y)) as parameter,
            and return True (False) if the lattice node is inside (outside) the
            region.
        """

        # sites belonging to the region
        s, pos = get_sites(self, location)

        N = self.scaling.density
        t = self.scaling.time
        vt = self.scaling.energy
        mu = self.scaling.mobility

        # default material parameters
        if self.input_length == 'm':
            mt = {'Nc': 1e25, 'Nv': 1e25, 'Eg': 1, 'epsilon': 1, 'mass_e': 1,\
              'mass_h': 1, 'mu_e': 100e-4, 'mu_h': 100e-4, 'Et': 0, 'tau_e': 1e-6,\
              'tau_h': 1e-6, 'affinity': 0, 'B': 0, 'Cn': 0, 'Cp': 0}
        else:
            mt = {'Nc': 1e19, 'Nv': 1e19, 'Eg': 1, 'epsilon': 1, 'mass_e': 1, \
                  'mass_h': 1, 'mu_e': 100, 'mu_h': 100, 'Et': 0, 'tau_e': 1e-6, \
                  'tau_h': 1e-6, 'affinity': 0, 'B': 0, 'Cn': 0, 'Cp': 0}

        arrays = {'Nc': self.Nc, 'Nv': self.Nv, 'Eg': self.Eg,\
                  'epsilon': self.epsilon, 'mass_e': self.mass_e,\
                  'mass_h': self.mass_h, 'mu_e': self.mu_e, 'mu_h': self.mu_h,\
                  'Et': self.Etrap, 'tau_e': self.tau_e, 'tau_h': self.tau_h,\
                  'affinity': self.bl, 'B': self.B, 'Cn': self.Cn, 'Cp': self.Cp}

        for key, val in mt.items():
            if key in mat.keys():
                val = mat[key]
            if not callable(val):
                arrays[key][s] = val
            else:
                arrays[key][s] = (val(pos) * location(pos))[s]

        # make values dimensionless
        self.Nc[s]     /= N
        self.Nv[s]     /= N
        self.Eg[s]     /= vt
        self.mu_e[s]   /= mu
        self.mu_h[s]   /= mu
        self.Etrap[s]  /= vt
        self.tau_e[s]  /= t
        self.tau_h[s]  /= t
        self.bl[s]     /= vt
        self.B[s]      /= (1./N)/t
        self.Cn[s]     /= (1./N**2)/t
        self.Cp[s]     /= (1./N**2)/t

        self.n1[s] = np.sqrt(self.Nc[s] * self.Nv[s]) * np.exp(-self.Eg[s]/2 + self.Etrap[s])
        self.p1[s] = np.sqrt(self.Nc[s] * self.Nv[s]) * np.exp(-self.Eg[s]/2 - self.Etrap[s])

        self.ni = np.sqrt(self.Nc * self.Nv) * np.exp(-self.Eg/2)

    def add_defects(self, location, N, sigma_e, sigma_h=None, E=None,
                    transition=(1,-1)):
        """
        Add additional charges (for a grain boundary for instance) to the total
        charge of the system. These charges are distributed on a line.

        Parameters
        ----------
        location: float or list of two array_like coordinates [(x1, y1), (x2, y2)] 
            Coordinate(s) in [cm] of a point defect or the two end points
            of a line defect.
        N: float or function
            Defect density of states [cm\ :sup:`-2` ]. Provide a float when the
            defect density of states is a delta function, or a function
            returning a float for a continuum. This function should take a
            single energy argument in [eV].
        sigma_e: float
            Electron capture cross section [cm\ :sup:`2`].
        sigma_h: float (optional)
            Hole capture cross section [cm\ :sup:`2`]. If not given, the same
            value as the electron capture cross section will be used.
        E: float
            Energy level of a single state defined with respect to the intrinsic
            Fermi level [eV]. Set to `None` for a continuum of states (default).
        transition: tuple
            Charge transition occurring at the energy level E.  The tuple (p, q)
            represents a defect with transition p/q (level empty to level
            occupied). Default is (1,-1).
        """

        if E is not None:
            E /= self.scaling.energy

        if isinstance(location, float):
            s, dl = utils.get_point_defects_sites(self, location)
        elif len(location) == 2:
            s, dl = utils.get_line_defects_sites(self, location)

        else:
            msg = "Wrong definition for the defects location: "\
            "the list must contain one number for a point, two points for a line, four points for a plane."
            logging.error(msg)
            return

        # The scale of the density of states is also the inverse of the scale 
        # for the capture cross section
        NN = self.scaling.density * self.scaling.length # m^-2
        if sigma_h is None:
            sigma_h = sigma_e
        sigma_e *= NN
        sigma_h *= NN

        if not callable(N):
            f = N / NN
        else:
            f = lambda E: N(E*self.scaling.energy) / NN

        params = defect(s, location, f, E, sigma_e, sigma_h, transition, dl)
        self.defects_list.append(params)

    def doping_profile(self, density, location):
        s, _ = get_sites(self, location)
        self.rho[s] = self.rho[s] + density / self.scaling.density

    def add_donor(self, density, location=lambda pos: True):
        """
        Add donor dopants to the system.

        Parameters
        ----------
        density: float
            Doping density [cm\ :sup:`-3`].
        location: Boolean function
            Definition of the region containing the doping. This function must
            take a tuple of real world coordinates (e.g. (x, y)) as parameters,
            and return True (False) if the lattice node is inside (outside) the
            region.
        """
        self.doping_profile(density, location)

    def add_acceptor(self, density, location=lambda pos: True):
        """
        Add acceptor dopants to the system.

        Parameters
        ----------
        density: float
            Doping density [cm\ :sup:`-3`].
        location: Boolean function
            Definition of the region containing the doping. This function must
            take a tuple of real world coordinates (e.g. (x, y)) as parameters,
            and return True (False) if the lattice node is inside (outside) the
            region.
        """
        self.doping_profile(-density, location)

    def generation(self, f, args=[]):
        """
        Distribution of generated carriers.

        Parameters
        ----------
        f: function
            Generation rate [cm\ :sup:`-3`].
        args: tuple
            Additional arguments to be passed to the function.
        """

        if callable(f):
            if f.__code__.co_argcount == 1 and self.ny==1:
                g = [f(x, *args) for x in self.xpts]
            else:
                g = [f(x, y, *args) for y in self.ypts for x in self.xpts]
        else:
            g = f

        self.g = np.asarray(g) / self.scaling.generation

        # compute the integral of the generation
        x = self.xpts / self.scaling.length
        y = self.ypts / self.scaling.length

        w = []
        u = []
        for j in range(self.ny):
            s = [i + j*self.nx for i in range(self.nx)]
            sp = spline(x, self.g[s])
            u.append(sp.integral(x[0], x[-1]))
        if self.ny > 1:
            sp = spline(y, u)
            w.append(sp.integral(y[0], y[-1]))
        if self.ny == 1:
            self.gtot = u[-1]
        else:
            self.gtot = w[-1]



    def contact_S(self, Scn_left, Scp_left, Scn_right, Scp_right):
        """
        Create the lists of recombination velocities that define the charge
        collection at the contacts out of equilibrium.

        Parameters
        ----------
        Scn_left: float
            Surface recombination velocity for electrons at the left contact [cm/s].
        Scp_left: float
            Surface recombination velocity for holes at the left contact [cm/s].
        Scn_right: float
            Surface recombination velocity for electrons at the right contact [cm/s].
        Scn_right: float
            Surface recombination velocity for electrons at the right contact [cm/s].

        """

        self.Scn = [Scn_left / self.scaling.velocity, 
                    Scn_right / self.scaling.velocity]
        self.Scp = [Scp_left / self.scaling.velocity, 
                    Scp_right / self.scaling.velocity]

    def contact_type(self, left_contact, right_contact, left_wf=None, right_wf=None):
        """
        Define the boundary conditions for the contacts at thermal equilibrium.
        'Ohmic' or 'Schottky' impose a value of the electrostatic potential,
        'Neutral' imposes a zero potential derivative.  

        Parameters
        ----------
        left_contact: string
            Boundary conditions for the contact at x=0.
        right_contact: string
            Boundary conditions for the contact at x=L.
        left_wf: float
            Work function for a Schottky contact at x=0.
        right_wf: float
            Work function for a Schottky contact at x=L.

        Notes
        -----
        Schottky contacts require work functions. If no values are given, an
        error will be raised.
        """

        # Schottky requires work functions
        if left_contact == 'Schottky' and left_wf is None:
            raise ValueError("Shcottky contacts require work functions.")
        if right_contact == 'Schottky' and right_wf is None:
            raise ValueError("Shcottky contacts require work functions.")

        self.contacts_bcs = [left_contact, right_contact]
        self.contacts_WF = [left_wf, right_wf]

def get_sites(sys, location):
    # find the sites which belong to a region
    nx, ny = sys.nx, sys.ny
    sites = np.arange(nx*ny, dtype=int)

    pos = np.transpose([np.tile(sys.xpts, ny), np.repeat(sys.ypts, nx)])
    if location.__code__.co_argcount == 1 and sys.ny==1:
        pos = pos[:,0]
    else:
        pos = (pos[:,0], pos[:,1])

    mask = location(pos)
    if type(mask) == bool:
        return sites, pos
    else:
        return sites[mask.astype(bool)], pos


